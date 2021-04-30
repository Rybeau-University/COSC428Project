import cv2
import math

INDICATOR_RADIUS = 40


class KeypointError(Exception):
    pass


def calculate_heatmap_colour(reference, current):
    difference = abs(reference - current)
    green = max([0, 255 - (difference * 20)])
    red = 10 * difference
    color = (0, green, red)
    return color


def draw_keypoints(frame, keypoints, analysis_dict=None, reference=None):
    for key in keypoints.keys():
        position = tuple([int(keypoints[key][0]), int(keypoints[key][1])])
        if (key == "left_shoulder" or key == "right_shoulder") and reference is not None:
            cv2.circle(frame, position, INDICATOR_RADIUS, calculate_heatmap_colour(reference["shoulders"],
                                                                                   analysis_dict["shoulders"]), cv2.FILLED)
        elif (key == "left_hip" or key == "right_hip") and reference is not None:
            cv2.circle(frame, position, INDICATOR_RADIUS, calculate_heatmap_colour(reference["hips"],
                                                                                   analysis_dict["hips"]), cv2.FILLED)
        else:
            cv2.circle(frame, position, INDICATOR_RADIUS, (0, 0, 255), cv2.FILLED)

    return frame


def calculate_tilt(lead, follow):
    y_change = lead[1] - follow[1]
    if y_change < 0:
        point_3 = tuple([int(follow[0]), int(lead[1])])
        angle = calculate_angle(abs(y_change), abs(lead[0] - follow[0]))
    elif y_change > 0:
        point_3 = tuple([int(lead[0]), int(follow[1])])
        angle = 0 - calculate_angle(abs(y_change), abs(lead[0] - follow[0]))
    else:
        angle = 0
        point_3 = None
    return angle


def calculate_lengths(angle_point, point_2):
    """
    Calculates the lengths of the side of the triangle needed to calculate the joint angles
    Returns longest side at index 0.
    """
    x_change = abs(angle_point[0] - point_2[0])
    y_change = abs(angle_point[1] - point_2[1])
    if angle_point[1] < point_2[1]:
        point_3 = [int(point_2[0]), int(point_2[1] - y_change)]
    else:
        point_3 = [int(angle_point[0]), int(angle_point[1] - y_change)]
    return tuple(point_3)


def calculate_analysis_dict(keypoints):
    analysis_dict = {
        "shoulders": calculate_tilt(keypoints["left_shoulder"], keypoints["right_shoulder"]),
        "hips": calculate_tilt(keypoints["left_hip"], keypoints["right_hip"])
    }
    return analysis_dict


def calculate_angle(opp, adjacent):
    """
    Calculates the angle of a right angle triangle
    """
    return math.degrees(math.atan((opp/adjacent)))


def angles_check(frame, predictions, reference, metadata):
    keypoints = get_keypoints(predictions, metadata)
    analysis_dict = calculate_analysis_dict(keypoints)
    vis_frame = draw_keypoints(frame, keypoints, analysis_dict, reference)
    return vis_frame


def create_reference(frame, predictions, metadata):
    keypoints = get_keypoints(predictions, metadata)
    analysis_dict = calculate_analysis_dict(keypoints)
    vis_frame = draw_keypoints(frame, keypoints)
    return vis_frame, analysis_dict


def get_keypoints(predictions, metadata):
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
            "left_shoulder": keypoints[keypoint_names.index('left_shoulder')],
            "right_shoulder": keypoints[keypoint_names.index('right_shoulder')]
        }
        return keypoint_dict
    else:
        raise KeypointError("Predictions has length 0")
