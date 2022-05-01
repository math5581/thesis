import configparser
from pyexpat import features
from TrackEval.trackeval import utils
from TrackEval import trackeval
from utilities.tracker.tracker import *
#import matplotlib.pyplot as plt
from utilities.utilities import *
from utilities.feature_extraction import *
from utilities.cv_utilities import *
from utilities.MOT import *


# This overwrites everything
save_vid = True
DIMENSIONS = (480, 480)
base_path = '/workspace/data/MOT17/train'

SEQUENCES = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN',
             'MOT17-05-FRCNN', 'MOT17-13-FRCNN']
#            'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN','MOT17-13-FRCNN']
SEQUENCES = ['MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN']

feature_extraction = FeatureDescription(DIMENSIONS)


def tracking_w_bbox(SEQUENCE):
    """ Current SOTA """
    dataloader = MOTDataloader(os.path.join(
        base_path, SEQUENCE), zebrafish=False)
    cv_tools = CVTools(DIMENSIONS)
    sequence_length = dataloader.get_seuqence_length()
    frame = dataloader.get_current_frame()

    tracker = Tracker(frame_width=frame.shape[1], frame_height=frame.shape[0])
    if save_vid:
        out = cv.VideoWriter(
            SEQUENCE+'.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, (frame.shape[1], frame.shape[0]))

    for i in range(sequence_length):
        #print('i ', i, ' of ', sequence_length)
        frame = dataloader.get_current_frame()
        # Get the set of current detections
        # dataloader.get_current_gt_bbox_scale()
        bbox_list = dataloader.get_current_det_bbox_scale()
        cv_tools.set_active_frame(frame)
        # bboxes = cv_tools.extract_bbox_from_list(bbox_list)
        frame_list = cv_tools.black_image_list_except_bbox(bbox_list)

        # Calculates the feature representation of each detection
        det_feat_arr = feature_extraction.get_detection_features(
            frames=frame_list, bbox_list=bbox_list)
        # Saves the extracted features and use them later on.
        #feature_extraction.save_detection_features(det_feat_arr, '/workspace/data/MOT17/eff_M_21k_ft1k_black/',
        #                                          SEQUENCE, dataloader.get_current_frame_id())

        #det_feat_arr = feature_extraction.load_detection_features('/workspace/data/MOT17/eff_M_21k_ft1k_blur_larger/',
        #                                           SEQUENCE, dataloader.get_current_frame_id(), bbox_list)
        # possibility to load here

        tracker.update_tracks(det_feat_arr)
        if save_vid:
            frame = tracker.draw_active_tracks_on_frame(frame)
        # Get integration?
        tracker.add_existing_tracks_to_df(dataloader.get_current_frame_id())
        if save_vid:
            out.write(frame)
        dataloader.next_frame()
    # save in
    path = os.path.join(base_path, 'trackers',
                        str(DIMENSIONS[0]), 'data')
    try:
        os.makedirs(path, exist_ok=False)
    except:
        pass
    out_file = os.path.join(path, SEQUENCE + '.txt')
    tracker.save_track_file(out_file)
    if save_vid:
        out.release()


def evaluation(seq, tracker=str(DIMENSIONS[0])):

    dataset = dataset_setup(tracker, seq)
    tracker = os.path.join(base_path, 'trackers', tracker)
    metrics_list, metric_names = metric_setup()

    def eval_sequence(seq, dataset, tracker, metrics_list, metric_names):
        """Function for evaluating a single sequence"""
        raw_data = dataset.get_raw_seq_data(tracker, seq)
        seq_res = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls='pedestrian')
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[met_name] = metric.eval_sequence(data)
        return seq_res
    result = eval_sequence(seq, dataset, tracker, metrics_list, metric_names)
    print('HOTA', result['HOTA']['HOTA(0)'])
    print('MOTA', result['CLEAR']['MOTA'])
    print('IDF1', result['Identity']['IDF1'])


def dataset_setup(tracker, seq):
    # Not really useful, but a necessity
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_dataset_config['SEQ_INFO'] = {seq: None}
    default_dataset_config['GT_FOLDER'] = base_path
    default_dataset_config['TRACKERS_FOLDER'] = os.path.join(
        base_path, 'trackers')
    default_dataset_config['TRACKERS_TO_EVAL'] = [tracker]
    default_dataset_config['DO_PREPROC'] = False  # disable
    dataset = trackeval.datasets.MotChallenge2DBox(default_dataset_config)
    return dataset


def metric_setup(metrics=['HOTA', 'CLEAR', 'Identity']):
    # Metric_setup
    default_metrics_config = {'METRICS': metrics, 'THRESHOLD': 0.5}
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in default_metrics_config['METRICS']:
            metrics_list.append(metric(default_metrics_config))
    metric_names = utils.validate_metrics_list(metrics_list)
    return metrics_list, metric_names


if __name__ == '__main__':
    # do more sequences
    for seq in SEQUENCES:
        print(seq)
        tracking_w_bbox(seq)
        evaluation(seq)
