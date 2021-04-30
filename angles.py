class KeypointError(Exception):
    pass

def angles_check(frame, predictions, reference, metadata):
    pass


def create_reference(frame, predictions, metadata):
    pass


def get_key_points(predictions, metadata):
    if len(predictions) > 0:
        keypoint_names = metadata.get("keypoint_names")
        keypoints = predictions.get('pred_keypoints').squeeze()
        keypoint_dict = {
            "left_wrist": keypoints[keypoint_names.index('left_wrist')],
            "right_wrist": keypoints[keypoint_names.index('right_wrist')],
            "left_elbow": keypoints[keypoint_names.index('left_elbow')],
            "right_elbow": keypoints[keypoint_names.index('right_elbow')],
            "left_knee": keypoints[keypoint_names.index('left_knee')],
            "right_knee": keypoints[keypoint_names.index('right_knee')],
            "left_hip": keypoints[keypoint_names.index('left_hip')],
            "right_hip": keypoints[keypoint_names.index('right_hip')],
            "left_shoulder": keypoints[keypoint_names.index('left_hip')],
            "right_shoulder": keypoints[keypoint_names.index('right_hip')]
        }
        return keypoint_dict
    else:
        raise KeypointError("Predictions has length 0")
