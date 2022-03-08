import cv2
import os

from cv2 import log
from logfile import logevent, logtext

def crop(ori_img_pth, img_pth, txt_pth, results_pth):
    """
        given an image, crop according to bbox, and save in output pth

        bbox = (x, y, width, height)

        saves image in data/results/cropped_images/
    """

    # logevent(f'cropping image {ori_img_pth}',1)
    output_folder = os.path.join(results_pth, 'cropped_images')
    if not os.path.isdir(output_folder):
        os.system(f'mkdir {output_folder}')

    # open txt file
    try:
        txtfile = open(txt_pth, 'r')
        txtcontents = txtfile.readlines()

        if len(txtcontents) == 1:
            txtcontents = txtcontents[0]

            content = txtcontents.split()
            x = int(content[0])
            y = int(content[1])
            w = int(content[2])
            h = int(content[3])
        else:
            logevent('no functionality for images with multiple detections', 4)

        img = cv2.imread(img_pth)
        crop_img = img[y:y+h, x:x+w]

        filename = os.path.split(img_pth)[1]

        cropped_path = os.path.join(output_folder, filename)
        cv2.imwrite(cropped_path, crop_img)

    except:
        logevent(f'crop.py: {img_pth} has no detections.',2)
    return output_folder