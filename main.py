# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


import argparse
import time
from sys import platform

from yolo.models import *
from yolo.utils.datasets import *
from yolo.utils.utils import *


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    classes = ["corner", "right", "straight", "sr", "slow", "ls", "left"]
    vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (832,832))
    # seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, n_init=10, max_iou_distance=0.9)
    results = []
    cap = cv2.VideoCapture('20180722110635_118.mp4')
     # Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    output = 'output'
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    cfg = 'cfg/yolov3.cfg'
    img_size = 832
    model = Darknet(cfg, img_size)
    
    # Load weights
    weights = 'weights/last.pt'
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    src = np.float32([[890,483], [1121,483], [634,700], [1381,695]])
    dst = np.float32([[800, 600], [1100, 600], [800, 900], [1100, 900]])
    p_matrix = cv2.getPerspectiveTransform(src, dst)
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]
    
    def frame_callback():
        # print("Processing frame %05d" % frame_idx)

        ret_val, img0 = cap.read()
        if img0 is None:
            cap.release()
            return -1
        
        img0 = cv2.warpPerspective(img0, p_matrix, (1920, 1080))
        img, *_ = letterbox(img0, new_shape=img_size)
        img = cv2.resize(img, (832, 832))

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        prediction, _ = model(img)
        # Load image and generate detections.
        # detections = create_detections(
        #     seq_info["detections"], frame_idx, min_detection_height)
        detections = []
        for image_i, pred in enumerate(prediction):
            
            class_conf, class_pred = pred[:, 5:].max(1)
            pred[:, 4] *= class_conf

            # Select only suitable predictions
            i = (pred[:, 4] > min_confidence) & (pred[:, 2:4] > 2).all(1) & torch.isfinite(pred).all(1)
            pred = pred[i]

            # If none are remaining => process next image
            if len(pred) == 0:
                continue

            # Select predicted classes
            class_conf = class_conf[i]
            class_pred = class_pred[i].unsqueeze(1).float()

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            pred[:, :2] = xywh2xyxy(pred[:, :4])[:, :2]
            pred[:, 2:4] *= 2
            # print(pred[:, :4])
            # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551
            class_pred_src = pred[:, 5:].cpu().detach().numpy()
            # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
            pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)
            # Get detections sorted by decreasing confidence scores
            # pred = pred[(-pred[:, 4]).argsort()]
            for pred_i in range(pred.shape[0]):
                det = Detection(pred[pred_i, :4].tolist(), pred[pred_i, 4], class_pred_src[pred_i])
                detections.append(det)
        
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        '''
        if display:
            # image = cv2.imread(
            #     seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            image = img0
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
        '''
        img, *_ = letterbox(img0, new_shape=img_size)
        img = cv2.resize(img, (832, 832))
        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            features = np.array(track.features)
            features = np.sum(features, axis=0)
            class_index = np.argmax(features)
            label = '%d %s' % (track.track_id, classes[class_index])
            plot_one_box(track.to_tlbr(), img, label=label, color=colors[track.track_id])
        #cv2.imshow('img', img)
        vid_writer.write(img)
        print(1)
        #cv2.waitKey(1)
        return 0

    # Run tracker.
    '''
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)
    '''
    while True:
        try:
            if frame_callback() < 0:
                break            
        except Exception as e:
            print(e)
            break
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    vid_writer.release()


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=False)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=False)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.5, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=0.1, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.5)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
