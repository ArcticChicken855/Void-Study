from pathlib import Path
import cv2
import numpy as np
import openpyxl

def calculate_void_percent(full_image):
    """
    Calculate the percentage of pixels in the image that are red.
    """
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red in HSV
    # Red can span over two ranges in the HSV space
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the two red ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Count the number of red pixels
    red_pixel_count = np.sum(red_mask > 0)

    # Calculate the total number of pixels in the image
    total_pixels = full_image.shape[0] * full_image.shape[1]

    # Calculate the percentage of red pixels
    percentage_red = (red_pixel_count / total_pixels) * 100

    return percentage_red

def main():
    """
    Go to the photoshopped images, and calculate the void percentage.
    """

    script_dir = Path(__file__).parent
    pre_cycle_dir = script_dir.parent / 'Non-code' / 'Void Study pre-cycle'
    post_cycle_dir = script_dir.parent / 'Non-code' / 'Void Study post-cycle'

    pre_images_full = dict()
    post_images_full = dict()
    
    # load the pre-cycle images into the dict
    for label_dir in pre_cycle_dir.rglob("*"):
        if label_dir.is_dir():

            for file_path in label_dir.rglob("*"):
                file_str = str(file_path).split("\\")
                filename = file_str[len(file_str)-1]

                if 'vP' in filename:
                    label = filename.split(".")[0].replace('vP', "")
                    pre_images_full[label] = cv2.imread(file_path)


    # load the post-cycle images into the dict
    for label_dir in post_cycle_dir.rglob("*"):
        if label_dir.is_dir():

            for file_path in label_dir.rglob("*"):
                file_str = str(file_path).split("\\")
                filename = file_str[len(file_str)-1]
                
                if 'vP' in filename:
                    label = filename.split(".")[0].replace('vP', "")
                    post_images_full[label] = cv2.imread(file_path)

    # Write the results to the excel file
    script_dir = Path(__file__).parent
    excel_file_path = script_dir.parent / 'Experimental Data' / 'Void Study FULL DOC.xlsx'
    excel_sheet = openpyxl.load_workbook(excel_file_path)

    for i, label in enumerate(pre_images_full.keys()):
        excel_sheet['Photoshop Void Data'].cell(row=1, column=i+1).value = label
        excel_sheet['Photoshop Void Data'].cell(row=2, column=i+1).value = calculate_void_percent(pre_images_full[label])

    for i, label in enumerate(post_images_full.keys()):
        excel_sheet['Post-Cycle Photoshop Void Data'].cell(row=1, column=i+1).value = label
        excel_sheet['Post-Cycle Photoshop Void Data'].cell(row=2, column=i+1).value = calculate_void_percent(post_images_full[label])

    excel_sheet.save(excel_file_path)

main()