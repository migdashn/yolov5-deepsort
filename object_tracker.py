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

def get_stgat_model(model_path, obs, pred):
    obs_len = obs
    pred_len = pred
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

def normalize_coordinates(coords, width, height, scale_factor=20):
    normalized = [(x / width * scale_factor, y / height * scale_factor) for x, y in coords]
    return normalized

def denormalize_coordinates(coords, width, height, scale_factor=20):
    denormalized = [(int(x / scale_factor * width), int(y / scale_factor * height)) for x, y in coords]
    return denormalized

def prepare_stgat_input(tracking_data, track_ids, width, height, obs, skip=10, scale_factor=20):
    obs_traj = []
    obs_traj_rel = []
    for track_id in track_ids:
        past_trajectory = tracking_data[track_id][-obs*skip:][::skip]
        past_trajectory = np.array(past_trajectory)
        normalized_traj = normalize_coordinates(past_trajectory[:, 1:], width, height, scale_factor)
        traj = torch.tensor(normalized_traj, dtype=torch.float32).cuda().unsqueeze(1)
        traj_rel = traj[1:] - traj[:-1]
        traj_rel = torch.cat([torch.zeros((1, 1, 2), dtype=torch.float32).cuda(), traj_rel], dim=0)
        obs_traj.append(traj)
        obs_traj_rel.append(traj_rel)
    obs_traj = torch.cat(obs_traj, dim=1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=1)
    return obs_traj, obs_traj_rel

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_

def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade, fde

def evaluate(pred_traj_gt, pred_traj_fake, seq_start_end):
    ade_outer, fde_outer = [], []
    total_traj = pred_traj_gt.size(1)

    ade, fde = [], []
    for _ in range(1):  # Only one sample for evaluation
        ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)
        ade.append(ade_)
        fde.append(fde_)
    ade_sum = evaluate_helper(ade, seq_start_end)
    fde_sum = evaluate_helper(fde, seq_start_end)

    ade_outer.append(ade_sum)
    fde_outer.append(fde_sum)

    ade = sum(ade_outer) / (total_traj * pred_traj_gt.size(0))
    fde = sum(fde_outer) / total_traj
    return ade, fde

def main(_argv):
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    obs = 8
    pred = 12
    skip = 5
    t_frames = obs * skip
    scale_factor = 20

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    model.eval()

    stgat_model = get_stgat_model(FLAGS.stgat_model, obs, pred)

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
    tracking_data = {}
    predicted_trajectories = {}
    trail_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
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
        trail_overlay.fill(0)  # Clear previous prediction lines

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

            # Save trail and coordinates for STGAT input
            centroid = (float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2))
            cv2.circle(trail_overlay, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)

            if track.track_id not in tracking_data:
                tracking_data[track.track_id] = []
            tracking_data[track.track_id].append((frame_num, centroid[0], centroid[1]))

            if track.track_id in predicted_trajectories:
                for j in range(1, len(predicted_trajectories[track.track_id])):
                    start_point = predicted_trajectories[track.track_id][j-1]
                    end_point = predicted_trajectories[track.track_id][j]
                    cv2.line(trail_overlay, start_point, end_point, (0, 255, 255), 5)  # Draw predicted trajectory in yellow

        if frame_num % t_frames == 0:
            track_ids = [track.track_id for track in tracker.tracks if track.track_id in tracking_data and len(tracking_data[track.track_id]) >= t_frames]
            if track_ids:  # Ensure track_ids is not empty
                print(f"Frame #{frame_num}: Track IDs ready for prediction: {track_ids}")
                try:
                    obs_traj, obs_traj_rel = prepare_stgat_input(tracking_data, track_ids, width, height, obs, skip, scale_factor)
                    seq_start_end = torch.tensor([(0, len(track_ids))], dtype=torch.int64).cuda()
                    pred_traj_fake_rel = stgat_model(obs_traj_rel, obs_traj, seq_start_end, 0, 3)
                    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    for i, track_id in enumerate(track_ids):
                        normalized_predicted_coords = [(coord[i][0].item(), coord[i][1].item()) for coord in pred_traj_fake]
                        predicted_coords = denormalize_coordinates(normalized_predicted_coords, width, height, scale_factor)
                        predicted_trajectories[track_id] = predicted_coords

                        if track_id == 5:    
                            print(f"Track ID {track_id} predicted trajectory:")
                            for coord in predicted_coords:
                                print(f"{coord[0]}, {coord[1]}")
                except Exception as e:
                    print(f"Error during prediction: {e}")

        frame = cv2.addWeighted(frame, 1, trail_overlay, 0.5, 0)

        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Collect all tracking data into a single list
    all_tracking_data = []

    for track_id, data in tracking_data.items():
        for entry in data:
            all_tracking_data.append((entry[0], track_id, entry[1], entry[2]))

    # Sort the collected data by frame first and then by track ID
    all_tracking_data.sort(key=lambda x: (x[0], x[1]))

    # Write the sorted data to the output file with float coordinates
    with open('tracking_data.txt', 'w') as f:
        for entry in all_tracking_data:
            f.write(f"{entry[0]} {entry[1]} {entry[2]:.2f} {entry[3]:.2f}\n")

    vid.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()

    # Evaluate the predictions
    # Filter out tracks where predicted trajectory length does not match ground truth length
    valid_track_ids = [track_id for track_id in predicted_trajectories.keys() if len([entry for entry in all_tracking_data if entry[1] == track_id]) >= pred]

    pred_traj_gt_list = [[[entry[2], entry[3]] for entry in all_tracking_data if entry[1] == track_id][-pred:] for track_id in valid_track_ids]
    pred_traj_fake_list = [[(coord[0], coord[1]) for coord in predicted_trajectories[track_id]] for track_id in valid_track_ids]
    # Ensure lists are not empty before creating tensors
    if pred_traj_gt_list and pred_traj_fake_list:
        pred_traj_gt = torch.tensor(pred_traj_gt_list).cuda()
        pred_traj_fake = torch.tensor(pred_traj_fake_list).cuda()

        # Ensure tensors are of shape (pred_len, num_tracks, 2)
        print(f"pred_traj_gt shape: {pred_traj_gt.shape}")
        print(f"pred_traj_fake shape: {pred_traj_fake.shape}")

        if len(pred_traj_gt.shape) == 3 and len(pred_traj_fake.shape) == 3:
            pred_traj_gt = pred_traj_gt.permute(1, 0, 2)
            pred_traj_fake = pred_traj_fake.permute(1, 0, 2)

            seq_start_end = torch.tensor([(0, pred_traj_fake.size(1))], dtype=torch.int64).cuda()

            ade, fde = evaluate(pred_traj_gt, pred_traj_fake, seq_start_end)

            # Output scores to the terminal and a file
            print(f"Dataset: zara2, Pred Len: {pred}, ADE: {ade:.12f}, FDE: {fde:.12f}")
            with open("evaluation.txt", "w") as f:
                f.write(f"Dataset: zara2, Pred Len: {pred}, ADE: {ade:.12f}, FDE: {fde:.12f}\n")
        else:
            print("Error: Tensors do not have the correct shape for permute")
    else:
        print("Error: Empty prediction lists. No valid tracks for evaluation.")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
