import os
import argparse 
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from absl import app, flags
from absl.flags import FLAGS
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

plt.style.use("seaborn-dark")
from STGAT.data.loader import data_loader
from STGAT.models import TrajectoryGenerator
from STGAT.utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('stgat_model', './STGAT/model_best.pth.tar', 'path to STGAT model checkpoint')


def get_stgat_model(model_path):
    # Parameters (assumed to be same as in STGAT's args)
    obs_len = 8
    pred_len = 12
    traj_lstm_input_size = 2
    traj_lstm_hidden_size = 32
    heads = [4, 1]
    hidden_units = [16]
    graph_network_out_dims = 32
    graph_lstm_hidden_size = 32
    noise_dim = (16,)
    noise_type = 'gaussian'
    dropout = 0.0
    alpha = 0.2
    
    # Initialize the model
    n_units = (
        [traj_lstm_hidden_size]
        + hidden_units
        + [graph_lstm_hidden_size]
    )
    model = TrajectoryGenerator(
        obs_len=obs_len,
        pred_len=pred_len,
        traj_lstm_input_size=traj_lstm_input_size,
        traj_lstm_hidden_size=traj_lstm_hidden_size,
        n_units=n_units,
        n_heads=heads,
        graph_network_out_dims=graph_network_out_dims,
        dropout=dropout,
        alpha=alpha,
        graph_lstm_hidden_size=graph_lstm_hidden_size,
        noise_dim=noise_dim,
        noise_type=noise_type,
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()
    return model

def prepare_stgat_input(tracking_data, track_ids, skip=5):
    obs_traj = []
    obs_traj_rel = []
    for track_id in track_ids:
        past_trajectory = tracking_data[track_id][-40:][::skip]  # Take the last 40 frames with a skip of 5 frames
        past_trajectory = np.array(past_trajectory)
        traj = torch.tensor(past_trajectory[:, 1:], dtype=torch.float32).cuda().unsqueeze(1) # Shape (seq_len, batch, 2)
        traj_rel = traj[1:] - traj[:-1] # Shape (seq_len-1, batch, 2)
        traj_rel = torch.cat([torch.zeros((1, 1, 2), dtype=torch.float32).cuda(), traj_rel], dim=0) # Prepend zeros to maintain the same length
        obs_traj.append(traj)
        obs_traj_rel.append(traj_rel)
    obs_traj = torch.cat(obs_traj, dim=1) # Shape (seq_len, batch, 2)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=1) # Shape (seq_len, batch, 2)
    return obs_traj, obs_traj_rel

def main(_argv):
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    model.eval()

    # Load STGAT model
    stgat_model = get_stgat_model(FLAGS.stgat_model)

    video_path = FLAGS.video

    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    trail_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    tracking_data = {}

    predicted_trajectories = {}

    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        print('Frame #: ', frame_num)
        start_time = time.time()

        results = model(frame)
        results = results.pandas().xyxy[0]
        results = results[results['name'] == 'person']

        bboxes = []
        scores = []
        for index, row in results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            score = row['confidence']
            bboxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(score)

        bboxes = np.array(bboxes)
        scores = np.array(scores)
        classes = np.array([0] * len(scores))

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, 'person', feature) for bbox, score, feature in zip(bboxes, scores, features)]

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            cv2.circle(trail_overlay, centroid, 5, (0, 0, 255), -1)

            # Save tracking data
            if track.track_id not in tracking_data:
                tracking_data[track.track_id] = []
            tracking_data[track.track_id].append((frame_num, centroid[0], centroid[1]))

            # Draw predicted trajectories
            if track.track_id in predicted_trajectories:
                for i in range(1, len(predicted_trajectories[track.track_id])):
                    start_point = predicted_trajectories[track.track_id][i-1]
                    end_point = predicted_trajectories[track.track_id][i]
                    cv2.line(frame, start_point, end_point, color, 2)

    # Predict trajectory every 40 frames for tracks with sufficient data
        if frame_num % 40 == 0:
            track_ids = [track.track_id for track in tracker.tracks if track.track_id in tracking_data and len(tracking_data[track.track_id]) >= 40]
            print(f"Frame #{frame_num}: Track IDs ready for prediction: {track_ids}")
            try:
                obs_traj, obs_traj_rel = prepare_stgat_input(tracking_data, track_ids, skip=5)
                seq_start_end = torch.tensor([(0, len(track_ids))], dtype=torch.int64).cuda()
                pred_traj_fake_rel = stgat_model(obs_traj_rel, obs_traj, seq_start_end, 0, 3)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                for i, track_id in enumerate(track_ids):
                    predicted_coords = [(int(coord[i][0].item()), int(coord[i][1].item())) for coord in pred_traj_fake]
                    predicted_trajectories[track_id] = predicted_coords

                    print(f"Track ID {track_id} predicted trajectory:")
                    for coord in pred_traj_fake[:, i, :]:
                        print(f"{coord[0].item()}, {coord[1].item()}")
            except Exception as e:
                print(f"Error during prediction: {e}")

        frame = cv2.addWeighted(frame, 1, trail_overlay, 0.5, 0)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save tracking data to file
    with open('tracking_data.txt', 'w') as f:
        for track_id, data in tracking_data.items():
            for entry in data:
                f.write(f"{track_id} {entry[0]} {entry[1]} {entry[2]}\n")

    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
