"""
    given the raw images, and relevant gt files, will train model
"""
import argparse
# from email.policy import default
import os
import json
from pickle import TRUE
from architecture.keypoint_regression.write_to_json import create_json
from training.training_new import training_model
from logfile import logevent

def parser():
    """
        command-line input arguments
    """
    parser = argparse.ArgumentParser(description = " Running algorithm")
    parser.add_argument("--training_dataset", type=str, default="", help="local location of folder that contains the raw images and gt annotations.")
    parser.add_argument("--no_of_keypnts", type=int, default=5, help="define whether you want to train for 3, 4 or 5 keypoints")
    parser.add_argument("--model_name", type=str, default="model1", help="name of model trained")
    return parser.parse_known_args()


def check_argument_errors(training_dataset_path, raw_images_pth, gt_files_pth):
    if not os.path.isdir(training_dataset_path):
        logevent(f'invalid training dataset path: {training_dataset_path}',4)
    if not os.path.isdir(raw_images_pth):
        logevent(f'no raw images pth in training dataset: {raw_images_pth}',4 )
    if not os.path.isdir(gt_files_pth):
        logevent(f'no gt files in training dataset: {gt_files_pth}', 4)

 
def main():
    args, unknownargs = parser()


    # load directories
    curr_dir = os.getcwd()
    training_dataset_path = os.path.join(curr_dir, args.training_dataset)
    raw_images_pth = os.path.join(training_dataset_path, 'raw_images')
    gt_files_pth = os.path.join(training_dataset_path, 'gt_keypnts')
    json_path = os.path.join(training_dataset_path, 'annotations.json')

    check_argument_errors(training_dataset_path, raw_images_pth, gt_files_pth)

    # if args.no_of_keypnts == 3 or args.no_of_keypnts==4:
    #     raise(Exception('need to add this functionality'))


    """
        create json
    """
    method = args.no_of_keypnts
    create_json(raw_images_pth, json_path, method, train=True)

    """
        run training
    """
    print(f'json_path: {json_path}')
    with open(json_path, 'r') as infile:
        json_data = json.load(infile)
    training_model(json_data, args.model_name, IMG_SIZE=224, BATCH_SIZE=64, NUM_KEYPOINTS=method,EPOCHS=10)



if __name__ == "__main__":
    main()