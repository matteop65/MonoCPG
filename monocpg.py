"""
    File run to run the entire model
"""

import argparse



def parser():
    """
        command-line input arguments
    """
    parser = argparse.ArgumentParser(description = " Running algorithm")
    parser.add_argument("--map", type=str, default="autoplex", help="This is which map is used. Autoplex or Town03 are the only supported a the moment.")
    parser.add_argument("--cam", type=int, default="8", help="This is which camera you want to use. This only works for AutopleX map. Only cameras 6 and 8 are installed at the moment.")
    parser.add_argument("--img_num", type=str, default="1.jpg", help="what type of image would you like to open. Only need a number, as directly automatically put to the relevant map.")
    parser.add_argument("--dataset", type=str, default="", help="Location of raw images. Typically data/raw_images")
    parser.add_argument("--method", type=int, default=5, help="Which method to use. 3, 4 or 5 anchor points")
    parser.add_argument("--keypnts", type=str, default='anchor_pts', help="this is where the keypoints are located (compared to predicted keypoitns)")
    return parser.parse_known_args()



if __name__=="__main__":
    
    
    """
        run 2D Object Detection
    """
