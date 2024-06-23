import os
import argparse
import time
import torch
import numpy as np
import cv2
from torch.backends import cudnn
import matplotlib.pyplot as plt
from absl import app, flags
from absl.flags import FLAGS
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from EfficientDet.backbone import EfficientDetBackbone
from EfficientDet.efficientdet.utils import BBoxTransform, ClipBoxes
from EfficientDet.utils.utils import preprocess, invert_affine, postprocess, preprocess_video, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

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

flags.DEFINE_string('video', 'https://manifest.googlevideo.com/api/manifest/hls_playlist/...', 'URL to input video stream')
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
        ade_, fde_ = cal_ade_fde(pred_traj_fake, pred_traj_gt)
        ade.append(ade_)
        fde.append(fde_)
    ade_sum = evaluate_helper(ade, seq_start_end)
    fde_sum = evaluate_helper(fde, seq_start_end)

    ade_outer.append(ade_sum)
    fde_outer.append(fde_sum)

    ade = sum(ade_outer) / (total_traj * pred_traj_gt.size(0))
    fde = sum(fde_outer) / total_traj
    return ade, fde

def adjust_classifier_for_pretrained(model, pretrained_num_classes, target_num_classes, compound_coef, anchor_ratios, anchor_scales):
    temp_model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=pretrained_num_classes, ratios=anchor_ratios, scales=anchor_scales)
    temp_model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'), strict=False)
    pretrain_weights = temp_model.classifier.header.pointwise_conv.conv.weight.data
    pretrain_bias = temp_model.classifier.header.pointwise_conv.conv.bias.data
    
    print(f"Shape of temp_model classifier weights: {pretrain_weights.shape}")
    print(f"Shape of temp_model classifier biases: {pretrain_bias.shape}")
    
    in_channels = model.classifier.header.pointwise_conv.conv.in_channels
    model.classifier.header.pointwise_conv.conv = torch.nn.Conv2d(in_channels, target_num_classes * len(anchor_ratios) * len(anchor_scales), kernel_size=(1, 1))
    model.classifier.header.pointwise_conv.bn = torch.nn.BatchNorm2d(target_num_classes * len(anchor_ratios) * len(anchor_scales))
    
    # Slice pretrain_weights to fit the model's new classifier
    model.classifier.header.pointwise_conv.conv.weight.data[:target_num_classes * len(anchor_ratios) * len(anchor_scales)] = pretrain_weights[:target_num_classes * len(anchor_ratios) * len(anchor_scales)]
    model.classifier.header.pointwise_conv.conv.bias.data[:target_num_classes * len(anchor_ratios) * len(anchor_scales)] = pretrain_bias[:target_num_classes * len(anchor_ratios) * len(anchor_scales)]

def main(_argv):
    FLAGS = flags.FLAGS
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    obs = 12
    pred = 16
    skip = 3
    t_frames = obs * skip
    scale_factor = 20
    max_frame_history = 60

    # Define anchor ratios and scales
    compound_coef = 0
    force_input_size = None 
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    threshold = 0.1
    iou_threshold = 0.1

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios, scales=anchor_scales)
    print("Loading model weights...")
    adjust_classifier_for_pretrained(model, pretrained_num_classes=90, target_num_classes=len(obj_list), compound_coef=compound_coef, anchor_ratios=anchor_ratios, anchor_scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'), strict=False)
    model.requires_grad_(False)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    stgat_model = get_stgat_model(FLAGS.stgat_model, obs, pred)

    video_path = FLAGS.video
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print("Error: Unable to open video stream")
        return

    out = None
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    else:
        out = None

    frame_num = 0
    tracking_data = {}
    predicted_trajectories = {}
    trail_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    prediction_overlay = np.zeros((height, width, 3), dtype=np.uint8)  # For permanent predictions

    while True:
        print("frame number : ", frame_num)
        return_value, frame = vid.read()
        if not return_value:
            print('Video has ended or failed, try a different video format!')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        start_time = time.time()
    
        ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

        if torch.cuda.is_available():
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        #print(f"Input Tensor Shape: {x.shape}")

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)

            out = invert_affine(framed_metas, out)

        detections = out[0]

        #print(f"Detections: {detections}")

        bboxes = []
        scores = []
        for i in range(len(detections['rois'])):
            #print(f"Detection entry: {detections['rois'][i]}")
            if detections['class_ids'][i] == 0:  # Only keep 'person' class
                bboxes.append(detections['rois'][i])
                scores.append(detections['scores'][i])

        bboxes = np.array(bboxes)
        if bboxes.size > 0:
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
        scores = np.array(scores)
        classes = np.array([0] * len(scores))

        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, 'person', feature) for bbox, score, feature in zip(bboxes, scores, features)]

        # Print the detections for debugging
        for detection in detections:
            #print(f"Detection: {detection}")
            #print(f"Bbox1: {detection.tlwh}, Confidence: {detection.confidence}, Class: {detection.class_name}")

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        if frame_num % max_frame_history == 0:
            trail_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            prediction_overlay = np.zeros((height, width, 3), dtype=np.uint8)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            print(f"Bbox2: {bbox}")
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
                    cv2.line(prediction_overlay, start_point, end_point, (0, 255, 255), 2)  # Draw predicted trajectory in yellow

        if frame_num % t_frames == 0:
            track_ids = [track.track_id for track in tracker.tracks if track.track_id in tracking_data and len(tracking_data[track.track_id]) >= t_frames]
            if track_ids:  # Ensure track_ids is not empty
                #print(f"Frame #{frame_num}: Track IDs ready for prediction: {track_ids}")
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
        frame = cv2.addWeighted(frame, 1, prediction_overlay, 0.5, 0)

        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        if out is not None and hasattr(out, 'write'):
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
    if out is not None and hasattr(out, 'release'):
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
