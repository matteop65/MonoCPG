""" 
    start the simulation
    Need to finish check_argument_errors()
"""

import enum
import statistics
import numpy as np
import argparse
import os
import time
from architecture.geometric_reasoning_algorithm.draw_bbox import anchor_points_4, anchor_points_5
from architecture.geometric_reasoning_algorithm.solve  import solve_main
from data.datasetv1.camera_information import camera_info

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
    return parser.parse_known_args()


def check_argument_error(args):
    """
        checks for arguments errors
    """
    # make sure the map is correct
    if args.map != "autoplex" or args.map != "town03":
        if args.map != "AutopleX" or args.map != "Autoplex":
            args.map = "autoplex"
        elif args.map != "Town03":
            args.map = "town03"
        else:
            raise(Exception("Map not available. Only maps available are autoplex and town03 for the time being."))
    return args    

def img_path(map, camera, img_num):
    """
        Returns img_path depending on map and camera
    """
    if map == "autoplex":
        if camera == 8:
            path = f"data/{map}/cam{camera}/{img_num}"
    
    return path


if __name__ == "__main__":
    args, unkownargs = parser()
    args = check_argument_error(args)

    print(f'\n------------------ Simulation Information ------------------')
    print(f'map: {args.map}')
    print(f'cam: {args.cam}')
    print(f'dataset: {args.dataset}')
    print(f'------------------------------------------------------------\n')

    # initialise variables
    camera = args.cam
    bbox_2D = [0]
    convention = np.array([[ 0,  1,  0 ], [ 0,  0, -1 ],[ 1,  0,  0 ]])
    image_path = img_path(args.map, args.cam, args.img_num)
    K, R_t, location = camera_info(args.map, args.cam)
    dimensions = []
    anchor_pt_1_location =[]
    img_num = 0

    # if a folder with raw images is provided, then go through this folder (and relevant one) to find the anchor points.
    if args.dataset:
        
        # make sure images folder exists
        MonoCPG = os.getcwd()
        results_folder = os.path.join(MonoCPG, 'data/results')
        if not os.path.isdir(results_folder):
            raise(Exception(f"images path not valid: {results_folder}"))
        
        images_folder = os.path.join(results_folder, 'images')
        # for each .jpg in image_folder, we want to run the solver.
        for dirpath, dirname, filenames in os.walk(images_folder):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    # find respective anchor pt file
                    keypoint_file = os.path.join(os.path.join(results_folder, 'global_keypnts'),os.path.splitext(filename)[0]+".txt")
                    # make sure keypoint_file exists
                    if not os.path.isfile(keypoint_file):
                        raise(Exception(f"anchot_pt information for {keypoint_file} does not exist.\n Looked for anchor_pt infor in {keypoint_file}"))


                    # extract the contents of the keypoint_file
                    no_of_anchor_pts = args.method                
                    txtfile = open(keypoint_file, "r")
                    txtcontents = txtfile.readlines()
                    anchor_pts = []
                    for row in txtcontents:
                        anchor_pts.append( list(map(int, row.split())))

                    # shorten the anchor_pts array to only pass through the same number as required by the solving method
                    # for solving method with 4 anchor points, only 4 anchor_pts should be passed through. 
                    anchor_pts = anchor_pts[:no_of_anchor_pts]
                    input = {
                        'image_folder':1,
                        'anchor_pts':anchor_pts,
                        'img_path': image_path,
                        'camera': camera,
                        'camera_location':location,
                        'intrinsics':K,
                        '2Dbbox': bbox_2D,
                        'number_of_anchor_points':no_of_anchor_pts,
                        'extrinsics':R_t,
                        'convention':convention
                    }

                    dim, anchor_info, vertices = solve_main(input)
                    dimensions.append(dim)

                    img_num += 1
                    print(f'------------------ RESULTS {img_num} ------------------')
                    print(f'image: {keypoint_file}')
                    print(f'solving method: {input["number_of_anchor_points"]} anchor points')
                    print(f'length: {dimensions[img_num-1][0]} m')
                    print(f'width:  {dimensions[img_num-1][1]} m')
                    print(f'height: {dimensions[img_num-1][2]} m')
                    # print(anchor_info)
                    print(f'------------------------------------------------\n')


                    # calculate the center of the vehicle
                    # for i, anchor_pt in enumerate(anchor_info):
                    #     for j,
                    # print(vertices)
                        # center length is 

                    # draw 3d bbox onto image
                    annotated_image_folder_path = os.path.join(results_folder,f"annotated_images_method_{no_of_anchor_pts}")
                    if not os.path.isdir(annotated_image_folder_path):
                        try:
                            print(f'[INFO] creating annotated image directory at: {annotated_image_folder_path}')
                            os.mkdir(annotated_image_folder_path)
                        except:
                            print(f"[WARNING] could not create directory, annotated_image_path {annotated_image_folder_path}")
                    annotated_image_path  = os.path.join(annotated_image_folder_path, filename)
                    
                    if os.path.isfile(annotated_image_path):
                        print(f'[WARNING] file, {annotated_image_path} already exists - renaming with UNIX time -.')
                        print(annotated_image_path[:-4])
                        os.rename(annotated_image_path, f'{annotated_image_path[:-4]}-{int(time.time())}.jpg')
                        

                    P = K @ convention @ R_t
                    P = np.concatenate([P, np.array((0, 0, 0, 1)).reshape(1,-1)], axis=0)
                   
                    if no_of_anchor_pts == 3:
                        raise(Exception("cannot draw 3d bbox for 3 anchor pnts yet"))
                        anchor_points_3(annotated_image_path, os.path.join(results_folder,"images",filename), P, vertices, dim)
                    elif no_of_anchor_pts == 4:
                        anchor_points_4(annotated_image_path, os.path.join(results_folder,"images",filename), P, vertices, dim)
                    elif no_of_anchor_pts == 5:
                        anchor_points_5(annotated_image_path, os.path.join(results_folder,"images",filename), P, vertices, dim)


    # for idx, num in enumerate(dimensions): print(dimensions[idx])

    average_dimensions = []
    median_dimensions = []

    for i in range(3):
        data = [item[i] for item in dimensions]
        average = statistics.mean(data)
        median = statistics.median(data)
        average_dimensions.append(average)
        median_dimensions.append(median)
    
    average_dimensions = np.round(average_dimensions, 4)
    median_dimensions = np.round(median_dimensions,4)
    print(f'mean dimensions:\t {average_dimensions}')
    print(f'median dimensions:\t {median_dimensions}')



    print('\n')
