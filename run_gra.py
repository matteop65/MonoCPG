""" 
    start the simulation
    Need to finish check_argument_errors()
"""

import enum
import codecs, json
from pprint import pprint
import statistics
from tkinter.ttk import Separator
import numpy as np
import argparse
import os
import time
from architecture.geometric_reasoning_algorithm.draw_bbox import keypoints_4, keypoints_5
from architecture.geometric_reasoning_algorithm.solve  import solve_main
from data.datasetv1.camera_information import camera_info
from logfile import logevent

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


def create_json(json_pth, img_pth, img_name, dim, delta_dim, keypoint_1, keypoint_2, keypoint_3, keypoint_4, keypoint_5, pi_1, pi_2, vertices, cam_location, distance_x_to_cam, distance_y_to_cam, end):
    keypoint_1["direction"] = keypoint_1["direction"].tolist()
    keypoint_2["direction"] = keypoint_2["direction"].tolist()
    keypoint_3["direction"] = keypoint_3["direction"].tolist()

    dictionary = {
        "img_path":img_pth,
        "img_name":img_name,
        "length":dim[0],
        "width":dim[1],
        "height":dim[2],
        "delta_length":delta_dim[0],
        "delta_width":delta_dim[1],
        "delta_height":delta_dim[2],
        "cam_location":cam_location, 
        "distance_x":distance_x_to_cam,
        "distance_y":distance_y_to_cam,
        "keypoint_1":keypoint_1,
        "keypoint_2":keypoint_2,
        "keypoint_3":keypoint_3
    }

    if keypoint_4 != None:
        keypoint_4["direction"] = keypoint_4["direction"].tolist()
        dictionary["keypoint_4"] = keypoint_4
    
    if keypoint_5!= None:
        keypoint_5["direction"] = keypoint_5["direction"].tolist()
        dictionary["keypoint_5"] = keypoint_5

    if pi_1 != None:
        dictionary["pi_1"] = pi_1
    
    if pi_2 != None:
        dictionary["pi_2"] = pi_2

    dictionary["vertices"]= [np.array(v).tolist() for v in vertices]

    pprint(dictionary)
    print(f'dictionary: {dictionary}')
    json_object = json.dumps(dictionary, indent=4)
    # json.dump(dictionary, codecs.open(json_pth, 'a+', encoding='utf=8'), separators=(",",":"), sort_keys=True, indent=4)

    with open(json_pth, 'a+') as f:
        # json.dumps(dictionary, f)
        f.write(json_object)
        if not end:
            f.write(",")


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


        # find the total number of images 
        total_images = 0   
        for dirpath, dirname, filenames in os.walk(images_folder):
            for filename in filenames:
                total_images += 1

        json_pth = os.path.join(results_folder,'outputs.json')

        # start json with [
        with open(json_pth, 'a+') as f:
            f.write("[")



        # for each .jpg in image_folder, we want to run the solver.
        for dirpath, dirname, filenames in os.walk(images_folder):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    img_pth = os.path.join(MonoCPG, args.dataset, "images", filename)

                    # find respective keypoint pt file
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
                    print(f'img_path: {img_pth}')
                    input = {
                        'image_folder':1,
                        'anchor_pts':anchor_pts,
                        'img_path': img_pth,
                        'camera': camera,
                        'camera_location':location,
                        'intrinsics':K,
                        '2Dbbox': bbox_2D,
                        'number_of_keypoints':no_of_anchor_pts,
                        'extrinsics':R_t,
                        'convention':convention
                    }

                    dim, keypoint_info, keypoint_vertices = solve_main(input)
                    dimensions.append(dim)

                    gt_dim = [7.3306, 2.326, 2.9739]
                    delta_dim = np.subtract(gt_dim, dim).tolist()

                    print(f'dim: {dim, dim[0]}')
                    # raise(Exception('h'))

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
                        keypoints_3(annotated_image_path, os.path.join(results_folder,"images",filename), P, vertices, dim)
                    elif no_of_anchor_pts == 4:
                        vertices = keypoints_4(annotated_image_path, os.path.join(results_folder,"images",filename), P, keypoint_vertices, dim)
                    elif no_of_anchor_pts == 5:
                        vertices = keypoints_5(annotated_image_path, os.path.join(results_folder,"images",filename), P, keypoint_vertices, dim)

                    print(f'vertices: {vertices}')
                    # save information into json
                    img_name = filename
                    length = dim[0]
                    width = dim[1]
                    height = dim[2]
                    keypoint_1 = keypoint_info["keypoint_1"]
                    keypoint_2 = keypoint_info["keypoint_2"]
                    keypoint_3 = keypoint_info["keypoint_3"]
                    try:
                        keypoint_4 = keypoint_info["keypoint_4"]
                    except:
                        logevent(f'could not find keypoint 4, this may be because solving procedure with 3 points was selected', 2)
                        keypoint_4 = None
                    try:
                        keypoint_5 = keypoint_info["keypoint_5"]
                    except:
                        logevent(f'could not find keypoint 5, this may be because solving procedure with 4 points was selected', 2)
                        keypoint_5 = None
                    try:
                        pi_1 = keypoint_info["pi_1"]
                        pi_2 = keypoint_info["pi_2"]
                    except:
                        logevent(f'could not find planes pi_1, pi_2. This may be because solving procedure with 5 points was not selected', 2)
                        pi_1, pi_2 = None

                    distance_x_to_cam = abs(location[0] - vertices[0][0]).tolist()
                    distance_y_to_cam = abs(location[1] - vertices[0][1]).tolist()
                    print(f'distance_x: {distance_x_to_cam}')


                    if img_num+1 == total_images:
                        create_json(json_pth, img_pth, img_name, dim, delta_dim, keypoint_1, keypoint_2, keypoint_3, keypoint_4, keypoint_5, pi_1, pi_2, vertices, location.tolist(), distance_x_to_cam[0], distance_y_to_cam[0], end=1)
                    else:
                        create_json(json_pth, img_pth, img_name, dim, delta_dim, keypoint_1, keypoint_2, keypoint_3, keypoint_4, keypoint_5, pi_1, pi_2, vertices, location.tolist(), distance_x_to_cam[0], distance_y_to_cam[0], end=0)                        
                    
                    img_num += 1
                    print(f'------------------ RESULTS {img_num} ------------------')
                    print(f'image: {img_pth}')
                    print(f'solving method: {input["number_of_keypoints"]} anchor points')
                    print(f'length: {dimensions[img_num-1][0]} m')
                    print(f'width:  {dimensions[img_num-1][1]} m')
                    print(f'height: {dimensions[img_num-1][2]} m')
                    print(f'------------------------------------------------\n')

    # finish ] at end of json
    with open(json_pth, 'a+') as f:
        f.write("]")
    

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
