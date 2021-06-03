"""
This file contains the code related to performing the joint angle calculation, comparison
and drawing of coloured markers on the frame.

Author: Ryan Beaumont (2021)
"""

import cv2
import math

"""Radius of the coloured markers"""
INDICATOR_RADIUS = 10


class KeypointError(Exception):
    """
    Custom Exception used for throwing exceptions specifically
    related to keypoint and joint angle calculation.
    """
    pass


def create_vector(point_1, point_2):
    """
    Returns the vector between two given points
    """
    return tuple([point_2[0] - point_1[0], point_2[1] - point_1[1]])


def output_angles(frame, analysis_dict, reference):
    """
    Returns the angles and the differences at the top left of the frame
    frame: Frame to apply text to
    analysis_dict: dictionary of the joint angles of interest from the amateur
    reference: dictionary of the joint angles of interest from the professional
    """
    y_pos = 20
    for key, value in analysis_dict.items():
        if key in reference.keys():
            text = "{}: Angle = {:.2f}, Diff = {:.2f}".format(key, value, value - reference[key])
            cv2.putText(frame, text, (0, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2,
                cv2.LINE_AA)
            cv2.putText(frame, text, (0, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y_pos += 20
    return frame


def calculate_heatmap_colour(reference, analysis):
    """
    Calculates the colour of the marker based upon the difference between the amateur
    and professional angle.
    reference: The joint angle of the professional
    analysis: The joint angle of the amateur
    """
    difference = abs(reference - analysis)
    green = max([0, 255 - (difference * 10)])
    red = 10 * difference
    color = (0, green, red)
    return color


def draw_keypoints(frame, keypoints, analysis_dict=None, reference=None):
    """
    Draws the coloured markers for the joint angles of interest onto the frame.
    Keypoints, analysis_dict and reference share key values, these are used to draw
    on the joint angles in the correct location.
    frame: Frame to draw on
    keypoints: the dictionary of joint locations in the frame
    analysis_dict: the dictionary of joint angles for the amateur
    reference: the dictionary of joint angles for the professional
    """
    for key in keypoints.keys():
        position = tuple([int(keypoints[key][0]), int(keypoints[key][1])])
        if (key == "left_shoulder" or key == "right_shoulder") and reference is not None:
            cv2.circle(frame, position, INDICATOR_RADIUS, calculate_heatmap_colour(reference["shoulders"],
                                                                                   analysis_dict["shoulders"]),
                       cv2.FILLED)
        elif (key == "left_hip" or key == "right_hip") and reference is not None:
            cv2.circle(frame, position, INDICATOR_RADIUS, calculate_heatmap_colour(reference["hips"],
                                                                                   analysis_dict["hips"]), cv2.FILLED)
        elif reference is not None and key in analysis_dict.keys() and key in reference.keys():
            cv2.circle(frame, position, INDICATOR_RADIUS, calculate_heatmap_colour(reference[key], analysis_dict[key]),
                       cv2.FILLED)
    return frame


def calculate_tilt(lead, follow):
    """
    Calculates the tilt of two joints in comparison to the horizontal plain.
    Depending on the direction of the tilt this is made negative to differentiate it.
    lead: leading shoulder position
    follow: following joint position
    """
    y_change = lead[1] - follow[1]
    if y_change < 0:
        angle = calculate_angle(abs(y_change), abs(lead[0] - follow[0]))
    elif y_change > 0:
        angle = 0 - calculate_angle(abs(y_change), abs(lead[0] - follow[0]))
    else:
        angle = 0
    return angle


def calculate_limb(angle_point, point_1, point_2):
    """
    Calculates the angle of a limb from the three points of that limb
    angle_point: the point where the angle is being calculated for
    point_1: One of the ends of the limb e.g. shoulder position
    point_2: Other end of the limb e.g. wrist position
    """
    vector_1 = create_vector(angle_point, point_1)
    vector_2 = create_vector(angle_point, point_2)
    angle = calculate_vector_angle(vector_1, vector_2)
    return angle


def calculate_analysis_dict(keypoints):
    """
    Creates the dictionary of joint angles from the dictionary of locations of joints in the frame.
    keypoints: the dictionary of joint locations in the frame
    """
    analysis_dict = {
        "shoulders": calculate_tilt(keypoints["left_shoulder"], keypoints["right_shoulder"]),
        "hips": calculate_tilt(keypoints["left_hip"], keypoints["right_hip"]),
    }
    if "left_hip" in keypoints.keys() and "left_knee" in keypoints.keys() and "left_ankle" in keypoints.keys():
        analysis_dict["left_knee"] = calculate_limb(keypoints['left_knee'], keypoints["left_hip"],
                                                    keypoints["left_ankle"])
    if "right_hip" in keypoints.keys() and "right_knee" in keypoints.keys() and "right_ankle" in keypoints.keys():
        analysis_dict["right_knee"] = calculate_limb(keypoints['right_knee'], keypoints["right_hip"],
                                                     keypoints["right_ankle"])
    if "right_shoulder" in keypoints.keys() and "right_elbow" in keypoints.keys() and "right_wrist" in keypoints.keys():
        analysis_dict["right_elbow"] = calculate_limb(keypoints['right_elbow'], keypoints["right_shoulder"],
                                                      keypoints["right_wrist"])
    if "left_shoulder" in keypoints.keys() and "left_elbow" in keypoints.keys() and "left_wrist" in keypoints.keys():
        analysis_dict["left_elbow"] = calculate_limb(keypoints['left_elbow'], keypoints["left_shoulder"],
                                                     keypoints["left_wrist"])
    return analysis_dict


def dot_product(v1, v2):
    """
    Returns the dot product of two vectors in R2
    """
    return v1[0] * v2[0] + v1[1] * v2[1]


def two_norm(v):
    """
    Returns the two-norm of a vector
    """
    return math.sqrt(dot_product(v, v))


def calculate_vector_angle(vector_1, vector_2):
    """
    Calculates the angle between two vectors using the cosine rule.
    The angle is made negative depending upon the direction of the bend
    to differentiate it from the same bend in the opposite direction.
    """
    dot = dot_product(vector_1, vector_2)
    cos_angle = float(dot / (two_norm(vector_1) * two_norm(vector_2)))
    # Buffer for floating point errors
    if 1.2 > cos_angle > 1:
        cos_angle = 1
    elif -1.2 < cos_angle < -1:
        cos_angle = -1
    elif -1.2 > cos_angle or 1.2 < cos_angle:
        raise KeypointError("Ratio for angle is outside of the domain.")
    if cos_angle > 0:
        multiplier = 1
    else:
        multiplier = -1
    angle_of_interest = (180 - math.degrees(math.acos(cos_angle))) * multiplier
    return angle_of_interest


def calculate_angle(opp, adjacent):
    """
    Returns the angle of interest in a right angle triangle using the opposite and adjacent sides
    opp: opposite side length
    adjacent: adjacent side length
    """
    return math.degrees(math.atan((opp / adjacent)))


def analyse_swing(frame, predictions, reference, metadata):
    """
    Performs the analysis and draws the visualisation for the amateur golfer.
    frame: Frame of the video to draw on
    predictions: Predictions output from Keypoint R-CNN
    reference: the dictionary of joint angles for the professional
    metadata: Metadata output from Keypoint R-CNN
    """
    keypoints = get_keypoints(predictions, metadata)
    analysis_dict = calculate_analysis_dict(keypoints)
    vis_frame = draw_keypoints(frame, keypoints, analysis_dict, reference)
    vis_frame = output_angles(vis_frame, analysis_dict, reference)
    return vis_frame


def create_reference(frame, predictions, metadata):
    """
    Creates the reference dictionary from the frame, predictions and metadata for the professional golfer.
    frame: Frame to draw on
    predictions: Predictions outputted from Keypoint R-CNN
    metadata: Metadata output from Keypoint R-CNN
    """
    keypoints = get_keypoints(predictions, metadata)
    analysis_dict = calculate_analysis_dict(keypoints)
    vis_frame = draw_keypoints(frame, keypoints)
    return vis_frame, analysis_dict


def get_keypoints(predictions, metadata):
    """
    Returns a dictionary of the joints and their positions in the frame.
    predictions: Predictions outputted from Keypoint R-CNN
    metadata: Metadata output from Keypoint R-CNN
    """
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
            "right_shoulder": keypoints[keypoint_names.index('right_shoulder')],
            "left_ankle": keypoints[keypoint_names.index('left_ankle')],
            "right_ankle": keypoints[keypoint_names.index('right_ankle')]
        }
        return keypoint_dict
    else:
        raise KeypointError("Predictions has length 0")
