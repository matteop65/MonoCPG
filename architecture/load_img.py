import cv2
import numpy as np


def mouse_callback(event, x, y, flags, anchor_info):
    """
        Allows for keypoints to be drawn with mouse"""
    anchor_pts = anchor_info[0]
    num_anchor_points = anchor_info[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(anchor_pts) < num_anchor_points:
            anchor_pts.append([x,y])
        else:
            anchor_pts.clear()

def load_img_place_anchor_pts(img_path, num_anchor_pts):
    """
        Loads an image for annotation. 
        Place anchor points on image. 
    """
    img = cv2.imread(f'{img_path}',1)
    anchor_pts = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback, [anchor_pts, num_anchor_pts])
    while True:
        img_copy = np.array(img)
        for p in anchor_pts:
            cv2.circle(img_copy, p, 2, [0,0,255],-1)

        cv2.imshow('image', img_copy)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            anchor_pts.clear()
    
    for i in range(len(anchor_pts)):
        print(f'anchor point {i+1}: {anchor_pts[i]}')

    return img, anchor_pts