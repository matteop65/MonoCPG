from asyncio import new_event_loop
from hashlib import new
from posixpath import expanduser
import cv2
import os 
import time
from logfile import logevent, logtext



def calculate_bbox(img, txt_contents):
    """
        given the txt contents, calculates new bbox

        returns: 
            - annotate image
            - new_top_left
            - new width
            - new height
    """
    # print(f'txtcontents {txt_contents}')
    info_array = txt_contents.split()
    vehicle_class = int(info_array[0])
    center_x = float(info_array[2])
    center_y = float(info_array[3])
    width = float(info_array[4])
    height = float(info_array[5])

    if vehicle_class == 7:
        # print(f'here2')
        img_height, img_width, channels = img.shape


        bbox_center_x = center_x*img_width
        bbox_center_y = center_y*img_height

        bbox_width = (width * img_width)
        bbox_height = (height * img_height)

        top_left = (int(bbox_center_x - bbox_width/2), int(bbox_center_y - bbox_height/2))
        top_right = (int(bbox_center_x+(bbox_width/2)), int(bbox_center_y-(bbox_height/2)))
        bottom_left = (int(bbox_center_x-bbox_width/2), int(bbox_center_y+bbox_height/2))
        bottom_right = ( int(bbox_center_x+bbox_width/2), int(bbox_center_y+bbox_height/2))

        # add 10 percent
        alpha = 1.1
        new_bbox_width = bbox_width * alpha
        new_bbox_height = bbox_height * alpha

        new_top_left = (int(bbox_center_x - new_bbox_width/2), int(bbox_center_y - new_bbox_height/2))
        new_top_right = (int(bbox_center_x+(new_bbox_width/2)), int(bbox_center_y-(new_bbox_height/2)))
        new_bottom_left = (int(bbox_center_x-new_bbox_width/2), int(bbox_center_y+new_bbox_height/2))
        new_bottom_right = ( int(bbox_center_x+new_bbox_width/2), int(bbox_center_y+new_bbox_height/2))

        # cv2.circle(img, (int(bbox_center_x), int(bbox_center_y)), 5, [0,0,255], -1)
        cv2.line(img, new_top_left, new_top_right, [0,0,255], 2)
        cv2.line(img, new_top_left, new_bottom_left, [0,0,255], 2)
        cv2.line(img, new_bottom_left, new_bottom_right, [0,0,255], 2)
        cv2.line(img, new_bottom_right, new_top_right, [0,0,255], 2) 

        return img, new_top_left, new_bbox_width, new_bbox_height
    else:
        return img, None, None, None


def saveimg(new_img_path, img):
    cv2.imwrite(new_img_path, img)

def savetxt(new_txt_path, msg):
    with open(new_txt_path, 'a+') as f:
        f.write(msg+'\n')

def increase_bbox(img_path, txt_path, filename, output, alpha):
    """
        This function increases the size of the 2D BBox output
        img_path: path to the raw images
        txt_path: path with the bbox files
        alpha: 1+percentage increase. For 10% increase alpha = 1.1
    """



    # define new images and txt files. 
    # txt files required for keypoint regression
    # img files should be raw, just a slightly nicer output
    output_folder = os.path.join(output, f'bbox_increase_{alpha}')
    if not os.path.isdir(output_folder):
        os.system(f'mkdir {output_folder}')

    # print(f'output_folder: {output_folder}')
    filename = os.path.splitext(filename)[0]
    new_img_path = os.path.join(output_folder, filename+'.jpg')
    new_txt_path = os.path.join(output_folder, filename+'.txt')

    # print(f'new_img_path: {new_img_path}')
    # print(f'new_txt_path {new_txt_path}')
    # if image file does not exist, return error
    if not os.path.isfile(img_path):
        logevent(f'image directory does not exist! {img_path}',4)

    # open image
    img = cv2.imread(img_path, 1)

    # open txt file
    txtfile = open(txt_path, 'r')
    txtcontents = txtfile.readlines()
    vehicle = []
    
    # print(f'txtcontents: {len(txtcontents)}')

    if len(txtcontents) == 1:
        img, new_top_left, new_bbox_width, new_bbox_height = calculate_bbox(img, txtcontents[0])

        # save img
        saveimg(new_img_path, img)
        # save txt
        savetxt(new_txt_path,f'{int(new_top_left[0])} {int(new_top_left[1])} {int(new_bbox_width)} {int(new_bbox_height)}' )
    else:
        # raise(Exception(f'txtcontents {txtcontents[:-1]}'))
        for vehicle in txtcontents[:]:    # want to remove the last element of txt contents as it is \n
            img, new_top_left, new_bbox_width, new_bbox_height = calculate_bbox(img, vehicle[:-1])

            if new_top_left != None:
                # print(f'{int(new_top_left[0])} {int(new_top_left[1])} {int(new_bbox_width)} {int(new_bbox_height)}')

                # save txt
                savetxt(new_txt_path,f'{int(new_top_left[0])} {int(new_top_left[1])} {int(new_bbox_width)} {int(new_bbox_height)}' )
            
                # save img
                saveimg(new_img_path, img)

                # raise(Exception('here'))
    return output_folder

if __name__ == "__main__":
    increase_bbox()