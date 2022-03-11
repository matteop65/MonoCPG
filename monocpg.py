"""
    File run to run the entire model
"""
import argparse
from email.policy import default
import shutil
from syslog import LOG_DEBUG
import time
import os

from pkg_resources import load_entry_point

from architecture.keypoint_regression.global_keypnts import transform_to_global_keypnts
from architecture.keypoint_regression.run_regression import run_regression
from architecture.object_detection_2d.increase_bbox import increase_bbox
from logfile import logevent, logtext
from architecture.crop import crop

"""
    Improvements:
        - outputs are coded in directly to results/...
"""

def parser():
    """
        command-line input arguments
    """
    parser = argparse.ArgumentParser(description = " Running algorithm")
    # parser.add_argument("--map", type=str, default="autoplex", help="This is which map is used. Autoplex or Town03 are the only supported a the moment.")
    # parser.add_argument("--cam", type=int, default="8", help="This is which camera you want to use. This only works for AutopleX map. Only cameras 6 and 8 are installed at the moment.")
    # parser.add_argument("--img_num", type=str, default="1.jpg", help="what type of image would you like to open. Only need a number, as directly automatically put to the relevant map.")
    parser.add_argument("--dataset", type=str, default="", help="Location of raw images. Typically data/raw_images")
    parser.add_argument("--method", type=int, default=5, help="Which method to use. 3, 4 or 5 anchor points")
    parser.add_argument("--keypnts", type=str, default='keypnts', help="this is where the keypoints are located (compared to predicted keypoitns)")
    parser.add_argument("--vgg_model_name", type=str, default='', help="what is the name of the trained keypoint regression model?")
    parser.add_argument("-v", action="store_true", default=False, help="just put -v if you want to visualise the keypoint outputs")
    return parser.parse_known_args()


def check_argument_errors(*args):
    """
        makes sure not errors with input arguments
    """
    if not os.path.isdir(dataset_path):
        raise(Exception(f'not valid directory for dataset: {dataset_path}'))
    # if not os.path.isdir(keypnts_path):
    #     raise(Exception(f'not valid keypnts input: {args.keypnts} as it points to path: {keypnts_path}'))
    if not os.path.isdir(darknet_path):
        logevent(f'could not find darknet path: {darknet_path}', 4)
    if method != 3 or method != 4 or method != 5:
        logevent(f'only support solving procedure of 3, 4 and 5 keypoints. You inserted {method} keypoints solving procedure')


if __name__=="__main__":
    # load arguments
    args, unknownargs = parser()

    # define parameters
    alpha=1.1

    # load directories
    curr_dir = os.getcwd()
    dataset_path = os.path.join(curr_dir, args.dataset)
    keypnts_path = os.path.join(curr_dir, dataset_path, args.keypnts)
    darknet_path = os.path.join(curr_dir, 'architecture/object_detection_2d/darknet')
    results_path = os.path.join(curr_dir, 'data/results')
    print(f'args.method: {args.method}')
    method = args.method
    check_argument_errors(dataset_path, keypnts_path, darknet_path, method)

    # raise(Exception(f"v{args.v}"))

    logevent(f'successfully loaded dataset: {dataset_path}',5)
    # logevent(f'successfully loaded keypnts path: {keypnts_path}',5)

    # load raw images path    
    raw_images_path = os.path.join(dataset_path, 'images')
    image_names = []
    for dirpath, dirname, filenames in os.walk(raw_images_path):
        image_names.append(filenames)
    image_names = image_names[0]
 
    #  print(f'raw_images_path: {raw_images_path}')

    """
        create results folder
    """
    if os.path.isdir(results_path):
        os.rename(results_path, f'{results_path}-{int(time.time())}')
    
    os.system(f'mkdir {results_path}')
    logevent(f'successfully created directory {results_path}', 5)

    """
        copy raw files into results folder
    """
    img_copy_fldr = os.path.join(results_path, 'images')
    if not os.path.isdir(img_copy_fldr):
        os.system(f'mkdir {img_copy_fldr}')
    for file in image_names:
        new_img_pth = os.path.join(img_copy_fldr, file)
        original_img_path = os.path.join(raw_images_path, file) 
        shutil.copyfile(original_img_path, new_img_pth)

    # raise(Exception('h'))
    """
        run 2D Object Detection
    """
    run_inference_dir = os.path.join(curr_dir, 'architecture/object_detection_2d')
    # command = f'cd {run_inference_dir} && python run_inference_yolov4.py --darknet_folder {darknet_path} '\
    #      f'--raw_files {raw_images_path} --stock'
    command = f'cd {run_inference_dir} && python run_inference_yolov4.py --darknet_folder {darknet_path} '\
         f'--raw_files {raw_images_path} --tl_folder {os.path.join(run_inference_dir,"YOLOv4-p6")} --weights best'
    exitcode = os.system(command)    
    if exitcode != 0:
        logevent(f"Inference did not run successfully (exit code {exitcode})",3)
    else:
        logevent(f'Succesffuly ran 2D Object Detection', 5)
    

    """
        increase size of bbox by 10%
    """
    cnt = 0
    logevent(f'Increasing size of bounding boxes',1)
    for file in image_names:
        img_pth = os.path.join(raw_images_path, file)
        txt_path = os.path.join(results_path,'Annotated-YOLOv4-p6_best', os.path.splitext(file)[0]+'.txt')
        increase_bbox_folder = increase_bbox(img_pth, txt_path, file, results_path, alpha)
    logevent(f'Succefully increased size of all bounding boxes by: {alpha}',5)

    """
        Crop images for keypoint regression    
    """
    logevent(f'Cropping images to only countain relevant bounding box.', 1)
    logevent(f'cropping image with red bbox. Please update this.', 2)

    YOLO_detections = []
    for dirpath, dirname, filenames in os.walk(os.path.join(results_path,'Annotated-YOLOv4-p6_best',)):
        YOLO_detections.append(filenames)
    YOLO_detections = YOLO_detections[0]

    for file in YOLO_detections:
        original_img_pth = os.path.join(raw_images_path, file)
        img_pth = os.path.join(results_path, f'bbox_increase_{alpha}/{os.path.splitext(file)[0]}.jpg')
        txt_path = os.path.join(results_path, f'bbox_increase_{alpha}/{os.path.splitext(file)[0]}.txt')

        cropped_folder = crop(original_img_pth, img_pth, txt_path, results_path)
        # new_img_path = os.path.join(results)
    logevent(f'Succeffully cropped all images in {increase_bbox_folder} to {cropped_folder}',5)
    """
        run keypoint regression, create json, scale keypnts
    """
    logevent(f'Running Keypoint Regression',1)
    cropped_folder = os.path.join(results_path, 'cropped_images')
    predicted_keypoints_folder, colours = run_regression(results_path, raw_images_path, cropped_folder, args.vgg_model_name, args.method ,args.v)
    logevent(f'Succesffuly run keypoint regression', 5)

    """
        Transpose local keypnts to global img coordinates
    """
    logevent(f'Transposing keypoints from cropped image to original dataset image',1)
    cropped_names = []
    for dirpath, dirname, filenames in os.walk(predicted_keypoints_folder):
        cropped_names.append(filenames)
    cropped_names = cropped_names[0]

    for file in cropped_names:
        bbox_2d_path = increase_bbox_folder
        local_keypoints_path = os.path.join(results_path, predicted_keypoints_folder)

        txt_name = os.path.splitext(file)[0] +'.txt'
        bbox_img_path = os.path.join(bbox_2d_path, txt_name)
        local_keypoint_txt = os.path.join(local_keypoints_path, txt_name)

        img_name = os.path.splitext(file)[0] +'.jpg'
        img_pth = os.path.join(raw_images_path, img_name)
        # logevent(f'predicted_keypoitns_folder {predicted_keypoints_folder}',2)
        # logevent(f'cropped names: {cropped_names}')
        # logevent(f'img_pth: {img_pth}')
        transform_to_global_keypnts(bbox_img_path, local_keypoint_txt, img_pth, colours) 

    logevent(f'Successfully transposed keypoints to global images', 5)

    """
        Run geometric reasoning algorithm
    """
    logevent(f'Running GRA', 1)
    os.system(f'python run_gra.py --dataset {dataset_path} --method {args.method}')
    logevent(f'Successfully run GRA', 5)