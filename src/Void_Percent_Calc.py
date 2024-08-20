from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares

def get_chip_contour(imsize, contours, max_size_ratio=None, chip_aspect_ratio=None, max_aspect_ratio_deviation = 0.1):
    """
    Find the largest contour that fufills the conditions.
    """
    if max_size_ratio is not None:
        contour_size_threshold = max_size_ratio * imsize

    largest_contour = contours[0]
    largest_contour_size = cv2.contourArea(largest_contour)

    for contour in contours:
        isChip = True
        cont_area = cv2.contourArea(contour)
        if cont_area > largest_contour_size:
            if max_size_ratio is not None: 
                if cont_area > contour_size_threshold:
                    isChip = False

            elif chip_aspect_ratio is not None:
                rect = cv2.minAreaRect(contour)
                rect_aspect_ratio = max(rect[1]) / min(rect[1])
                if (rect_aspect_ratio < (1-max_aspect_ratio_deviation*chip_aspect_ratio)) or (rect_aspect_ratio > (1+max_aspect_ratio_deviation*chip_aspect_ratio)):
                    isChip = False
        else:
            isChip = False

        if isChip is True:
            largest_contour = contour
            largest_contour_size = cv2.contourArea(contour)

    return largest_contour

def isolate_chip_region(image, dispBool=False, dispRect=False, max_size_ratio=None, chip_aspect_ratio=None):

    """
    Take in a 2D xray of a rectangular chip. Rotate the image so that the chip is straight.
    Return the rotated image, the corners of the rectangle, and a numpy array of the pixel vals in the rectangle.
    """

    # Convert to boolean image
    mean = np.mean(image)
    _, boolean_im = cv2.threshold(image, mean*1.0, 255, cv2.THRESH_BINARY)
    if dispBool is True:
        cv2.imshow('Boolean Image', boolean_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(boolean_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # get the contour that covers the chip area
    chip_contour = get_chip_contour(image.shape[0] * image.shape[1], contours, max_size_ratio, chip_aspect_ratio)

    # get the smallest bounding rectangle for the contour
    rect = cv2.minAreaRect(chip_contour)

    # Get the rotation matrix and rotate the image so that the rectangle is straight
    width = round(rect[1][0])
    height = round(rect[1][1])
    angle = rect[2]

    rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1.0)

    rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    rot_rect = (rect[0], rect[1], 0)

    # Convert the rectangle to four corner points
    box = cv2.boxPoints(rot_rect)
    intbox = np.array(box, dtype=np.int64)
    for i, point in enumerate(box):
        intbox[i, 0] = round(point[0])
        intbox[i, 1] = round(point[1])

    if dispRect is True:
        cv2.polylines(rotated, [intbox], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.imshow('Image with Rectangle', rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # convert the rectangular chip region into a numpy array
    p1 = intbox[0]
    p3 = intbox[2]
    x1 = min(p1[0], p3[0])
    x2 = max(p1[0], p3[0])
    y1 = min(p1[1], p3[1])
    y2 = max(p1[1], p3[1])
    
    chip = rotated[y1:y2, x1:x2]

    return rotated, intbox, chip

def detect_voids(chip):
    """
    Given the isolated chip region, detect voids and report the void percentage.
    """



def main():
    # Load the image as a grayscale
    script_dir = Path(__file__).parent
    label = 'C5'
    image_file_path = script_dir.parent / 'Non-code' / 'Void Study pre-cycle' / label[0] / f'{label}.jpg'
    grey_im = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

    # Isolate the chip region
    aspect_ratio = 1.264
    rotated_im, box, chip = isolate_chip_region(grey_im, dispRect=True, max_size_ratio=0.8, chip_aspect_ratio=aspect_ratio)

    # Detect voids within the chip region
    void_percentage = detect_voids(chip)


main()
