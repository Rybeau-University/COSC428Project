import cv2
import numpy as np

# Here are the HSV colour values used by the inRange function for detecting the three balls.
# The red color runs a hue from 170 to 5, so we need to have two thresholds to catch that rollover.
lo_red = np.array([170,150,200])
hi_red = np.array([179,255,255])
lo_red2 = np.array([0,150,200])
hi_red2 = np.array([5,190,255])

lo_green = np.array([35,100,0])
hi_green = np.array([50,255,255])

lo_orange = np.array([10,140,200])
hi_orange = np.array([34,255,255])

kernel = np.ones((3,3), np.uint8) # Morphological kernel

def morph_mask(mask):
    # Morphological operations for cleaning up the mask from the inRange function.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask

def get_contour(mask, color):
    # Find the contour of the image mask from inRange.
    MIN_VALID_CONTOUR_SIZE = 100  # The minimum size of a valid contour. Used to eliminate small false-positives.
    max_size = 0
    valid_index = -1  # In this case, an index of -1 means that no contour has been found.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check over the detected contours to find the largest one that passes the threshold set in MIN_VALID_CONTOUR_SIZE.
    for i in range(len(contours)):
        latest_contour_size = cv2.contourArea(contours[i])
        if latest_contour_size > max_size and latest_contour_size > MIN_VALID_CONTOUR_SIZE:
            max_size = latest_contour_size
            valid_index = i

    # If no contours were found, or none that were sufficiently large, return None for the location and radius.
    # Otherwise, return the center point and radius of the minimum enclosing circle of the contour.
    if len(contours) == 0 or valid_index == -1:
        x = None
        y = None
        radius = None
        return (None, None), None
    else:
        (x,y),radius = cv2.minEnclosingCircle(contours[valid_index])
        return (int(x),int(y)), int(radius)
        

def get_ball_coordinates(hsv):
    # Determine the center point and radius of the three balls.
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lo_red, hi_red), cv2.inRange(hsv, lo_red2, hi_red2))  # Bitwise OR to combine the masks
    red_mask = morph_mask(red_mask)

    green_mask = cv2.inRange(hsv, lo_green, hi_green)
    green_mask = morph_mask(green_mask)

    orange_mask = cv2.inRange(hsv, lo_orange, hi_orange)
    orange_mask = morph_mask(orange_mask)

    red_center,red_radius = get_contour(red_mask, "red")
    green_center,green_radius = get_contour(green_mask, "green")
    orange_center,orange_radius = get_contour(orange_mask, "orange")

    return red_center,red_radius,green_center,green_radius,orange_center,orange_radius

def test_ball_dropped(ball_center, left_wrist, right_wrist, dropped_ball_threshold):
    def ball_below_wrist(ball, wrist, threshold):
        # Check that the distance is greater than the threshold, and that the ball is lower than the wrist in the frame
        return np.sqrt((ball[0] - wrist[0])**2 + (ball[1] - wrist[1])**2) > threshold and ball[1] > wrist[1]
    
    if ball_below_wrist(ball_center, left_wrist, dropped_ball_threshold) or ball_below_wrist(ball_center, right_wrist, dropped_ball_threshold):
        return True
    else:
        return False