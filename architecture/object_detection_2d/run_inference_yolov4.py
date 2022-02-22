"""
    This program will run yolov4 automatically by inputting just video or image dir.
    General format:
        python3 run_inference_yolov4.py -darknet_folder= -tl_folder= -tl_folder= -weights= -video
    If want to use yolo darknet format:
        python3 run_inference_yolov4.py -darknet_folder= -darknet_format='./darknet detector...'
    Only advantage of running the traditional darknet_format from this file is that it can be run not being within
        darknet directory.
    Videos can be run as batch. Just point to video directory.
"""

"""
    darknet format for images:
        ./darknet detector test <labels> <cfg_file> <weights> -dont_show -ext_output -avgframes 1 -save_labels_with_confidence <input images>
        e.g.:
        ./darknet detector test cfg/coco.data cfg/yolov4-p6.cfg weights/yolov4-p6.weights -dont_show -ext_output -avgframes 1 -save_labels_with_confidence ~/Pictures/Raw_Frames

    darknet format for videos:
        ./darknet detector demo <labels> <cfg_file> <weights> <input video> -out <output_name>.json -out_filename <output_name>.mp4
        e.g.:
        ./darknet detector demo ./cfg/coco.data ./cfg/yolov4-p6.cfg ./weights/yolov4-p6.weights ~/Videos/Cam1_1sec_Sample.mp4 -out Cam1_1sec_Output_cust.json -out_filename Cam1_1sec_Output_cust.mp4 -ext_output -avgframes 1
"""

import os
import argparse
import time
import shutil
import warnings
from logfile import logtext,logevent


def parser():
    """
        Argument inputs for darknet_folder, raw_files and tl_folder need to start with ~/.
        If you want to annotate a video, make sure to have either the --video
    """
    parser = argparse.ArgumentParser(description="Running the training algorithm")
    parser.add_argument('--darknet_folder', type=str, default='', help='darknet folder directory')
    parser.add_argument("--raw_files", type=str, default="", help="raw files directory")
    parser.add_argument('--tl_folder', type=str, default='', help='transfer learning folder. must contain weights and '
                                                                 'cfg file with the same name as the folder. '
                                                                 'And obj.data. ')
    parser.add_argument('--video_out', type=str, default='', help='where you want to put your annotated videos. Not a compulsory '
                                                                    'argument. Put -1 if you want annotated files in the same directory as '
                                                                    'raw video files')
    parser.add_argument('--weights', type=str, default='best', help='just put end of weights file e.g. final')
    parser.add_argument('--darknet_format', type=str, default='', help='input darknet format here e.g. '
                                                                      './darknet detector test...')
    parser.add_argument('--stock', action='store_true', default='', help='if you want to run stock yolov4-p6')
    return parser.parse_known_args()


def check_argument_errors(args):
    if args.darknet_folder:
        if not os.path.isdir(os.path.expanduser(args.darknet_folder)):
            raise (ValueError('Invalid darknet directory: {}'.format(os.path.expanduser(args.darknet_folder))))

    if not args.darknet_format:
        if not args.stock:
            if not os.path.isdir(os.path.expanduser(args.tl_folder)):
                raise (ValueError("Invalid transfer learning folder: {}".format(os.path.expanduser(args.tl_folder))))

        if args.video_out:
            if not os.path.isfile(os.path.expanduser(args.raw_files)):
                if not os.path.isdir(os.path.expanduser(args.raw_files)):
                    raise (ValueError('Invalid video or video folder path: {}'.format(os.path.expanduser(args.raw_files))))
        else:
            if not os.path.isfile(os.path.expanduser(args.raw_files)):
                if not os.path.isdir(os.path.expanduser(args.raw_files)):
                    raise (ValueError("Invalid raw image or folder path: {}".format(os.path.expanduser(args.raw_files))))
    else:
        logevent('Cannot check for input errors for darknet_format',2)


def copy_annotated_video(raw_files_dir, annotated_file_name, tl_folder, darknet_path, weights, video_out):
    """
        YOLOv4 will naturally leave the annotated video in the darknet directory. 
        This function will copy over that video to a folder next to the raw video files. 
        The json files generated whilst running darknet are also copied over to the same location. 
    """

    raw_video_root_dir = os.path.split(raw_files_dir)[0]
    tl_name = os.path.split(tl_folder)[1]

    # original locations variables
    video_name = annotated_file_name
    if video_name.endswith('.mp4'):
        video_name = video_name[0:len(video_name)-4]
        file_extension = 'mp4'

    original_video_path = os.path.join(darknet_path, f"{video_name}.{file_extension}")

    # new locations variables
    if tl_name =='yolov4-p6':
        new_video_name = f'{video_name}-{tl_name}.{file_extension}'
    else:
        new_video_name = f'{video_name}-{tl_name}_{weights}.{file_extension}'
    
    if video_out=='-1':
        new_folder = os.path.join(raw_video_root_dir, f'Annotated_video_{video_name}-{tl_name}_{weights}')
        new_video_path = os.path.join(new_folder, new_video_name)
    else:
        print(f'_______here_____ video_out: {video_out}')
        new_folder = os.path.join(video_out, f'Annotated_video_{video_name}-{tl_name}_{weights}')
        print(f'____herenot____ new_folder = {new_folder}')
        new_video_path = os.path.join(new_folder, new_video_name)

    # need to copy over json as well. .mp4 and .json files have same names, so can replace the file extension. 
    if file_extension == 'mp4':
        original_json_path = original_video_path[0:len(original_video_path)-4]+'.json'
        new_json_path = new_video_path[0:len(new_video_path)-4]+'.json'

    # folder with some annotated videos may already exist. Read in folder and see whether this exact file already exists. 
    if os.path.isdir(new_folder):
        files = [f for f in os.listdir(new_folder) if os.path.isfile(os.path.join(new_folder,f))]
        for f in files:
           # if annotated video file already exists, rename existing with appended UNIX time.
            if f == new_video_name:
                logevent(f'File {new_video_name} copying over to: {new_folder} already exists!\nWill rename previous '
                    'file with appended UNIX time',2)
                os.rename(new_video_path, f'{new_video_path}-{int(time.time())}')
            
            if f == os.path.split(new_json_path)[1]:
                logevent(f'File {new_json_path} copying over to: {new_folder} already exists!\nWill rename previous '
                    'file with appended UNIX time',2)
                os.rename(new_json_path, f'{new_json_path}-{int(time.time())}')
    else:
        logevent(f'Creating folder at {new_folder}',1)
        os.system(f'mkdir {new_folder}')

    video_path_dir = os.path.split(new_video_path)[1]
    if not os.path.isdir(video_path_dir):
        os.system(f'mkdir {video_path_dir}')
    logevent(f'Video_path_dir = {video_path_dir}',1)
    logevent(f'video_name = {video_name}')
    logevent(f'tl_name = {tl_name}')
    logevent(f'Original_video_path = {original_video_path}',1)
    logevent(f'new_video_path = {new_video_path}',1)
    shutil.copyfile(original_video_path, new_video_path)
    shutil.copyfile(original_json_path, new_json_path)     

    # once file is copied remove original
    logevent(f'Original file at {original_video_path} and {original_json_path} have been removed and copied to new path {new_video_path} and {new_json_path}',1)
    os.remove(original_json_path)
    os.remove(original_video_path)  
        

def return_annotated_directory(raw_files_dir, tl_folder, weight_type):
    """
        As YOLOv4 overwrites all the raw image files.
        This function will copy over all the raw image files to an YOLOv4-annotated-{dataset_name} folder.
        This new folder will be passed directly to YOLOv4 to be overwritten.
    """

    # folder with annotated images will be created in the same parent directory as where the raw files are present.
    images_root_dir = os.path.split(raw_files_dir)[0]
    target_annotations_folder = os.path.split(tl_folder)[1]
    # Don't include YOLOv4 if target_annotations_folder already contains it
    if "YOLOv4".lower() in target_annotations_folder.lower():
        annotated_images = os.path.join(images_root_dir, f'Annotated-{target_annotations_folder}_{weight_type}')
    else:
        annotated_images = os.path.join(images_root_dir, f'Annotated-YOLOv4-{target_annotations_folder}_{weight_type}')

    # if folder for annotated images already exists, then rename folder to append UNIX time, as to not overwrite.
    if os.path.isdir(annotated_images):
        os.rename(annotated_images, f'{annotated_images}-{int(time.time())}')
    logevent(f"Source images: {raw_files_dir}",1)
    logevent(f"Destination images: {annotated_images}",1)

    if os.path.isdir(raw_files_dir):
        shutil.copytree(raw_files_dir, annotated_images)
    elif os.path.isfile(raw_files_dir):
        annotated_images = annotated_images + '-' + os.path.split(raw_files_dir)[1]
        shutil.copyfile(raw_files_dir, annotated_images)

    logevent(f'annotated_images: {annotated_images}',1)
    logevent(f'raw_files_dir: {raw_files_dir}',1)
    return annotated_images


def run_yolov4(raw_files, cfg, weights, coco_data, darknet_path, video):
    """
        Cd's into darknets directory and calls YOLOv4 with the arguments it expects.
    """
    annotated_file=''
    if video:
        filename = os.path.split(raw_files)[1]
        if filename.endswith('.mp4'):
            len_filename = len(filename)
            # output the annotated video and the json with annotations as the same name. 
            output_json = filename[0:len_filename - 4] + '-annotated' + '.json'
            annotated_file = filename[0:len_filename - 4] + '-annotated' + '.mp4'
        # what if I change the output directory from here? test on monday
        inferenceCmd = f'cd "{darknet_path}" && ./darknet detector demo "{coco_data}" "{cfg}" "{weights}" "{raw_files}" -out "{output_json}" -out_filename {annotated_file} -ext_output -avgframes 1'
        logevent(f"Executing: {inferenceCmd}",1)
        exitcode = os.system(inferenceCmd)
        if exitcode != 0:
            logevent(f"Inference did not run successfully (exit code {exitcode})",3)

    else:
        logevent(f'raw_files: {raw_files}',1)
        inferenceCmd = f'cd "{darknet_path}" && ./darknet detector test "{coco_data}" "{cfg}" "{weights}" -dont_show -ext_output -avgframes 1 -save_labels_with_confidence "{raw_files}"'
        logevent(f"Executing: {inferenceCmd}",1)
        exitcode = os.system(inferenceCmd)
        if exitcode != 0:
            logevent(f"Inference did not run successfully (exit code {exitcode})",3)

    return annotated_file

def get_video_list(video_folder):
    video_files_list = list()
    for video in [f for f in os.listdir(video_folder) if f.endswith('.mp4')]:
        video_files_list.append(f'{os.path.join(video_folder, video)}')
    return video_files_list


def check_for_video_batch(video_folder):
    """
        YOLOv4 does not run videos as batch. 
        This function is a hackish way of running videos as a batch. 
        Use the --raw_files argument to point to the folder with videos. This function returns the list of videos. 
        Then run_yolov4 is run on a for loop for each video in the folder. 
    """
    logevent(f'video folder: {video_folder}',1)
    results = [x[0] for x in os.walk(video_folder)] if not video_folder.endswith('.mp4') else False
    logevent(f'results: {results}',1)
    if results:
        videos_list = get_video_list(video_folder)
        return True, videos_list
    else:
        return False, video_folder
        


def main():
    args,unknownargs = parser()
    check_argument_errors(args)
    
    darknet_path = os.path.expanduser(args.darknet_folder) if args.darknet_folder else False
    video_out = os.path.expanduser(args.video_out)


    if not args.darknet_format:
        raw_files_dir = os.path.expanduser(args.raw_files)

        if args.stock:
            cfg = os.path.join(darknet_path, 'cfg/yolov4-p6.cfg')
            weights = os.path.join(darknet_path, 'weights/yolov4-p6.weights')
            coco_data = os.path.join(darknet_path, 'cfg/coco.data')
            tl_folder=os.path.join(darknet_path, 'yolov4-p6')
        else:
            # Remove trailing slash (if exists)
            tl_folder_trailing = os.path.expanduser(args.tl_folder)
            tl_folder_YOLO = os.path.normpath(tl_folder_trailing)
            tl_folder = tl_folder_YOLO.replace('YOLOv4-', '')

            logevent(f"TL Folder: {os.path.split(tl_folder)}",1)
            cfg = os.path.join(tl_folder_YOLO, 'yolov4-'+os.path.split(tl_folder)[1] + '.cfg')
            if not args.weights:
                weights = os.path.join(tl_folder_YOLO, 'yolov4-'+os.path.split(tl_folder)[1] + '.weights')
            else:
                weights = os.path.join(tl_folder_YOLO, 'yolov4-'+os.path.split(tl_folder)[1] + f'_{args.weights}.weights')
            coco_data = os.path.join(tl_folder_YOLO, 'obj.data')
            logevent(f'Obj data {coco_data}',1)
        
        video =1 if video_out else False

        if video:
            video_batch, video_list = check_for_video_batch(raw_files_dir)
            if video_batch:
                for video in video_list:
                    annotated_file_names = run_yolov4(video, cfg, weights, coco_data, darknet_path, video)
                    copy_annotated_video(video, annotated_file_names, tl_folder_YOLO, darknet_path, args.weights, video_out)
            else:                   
                annotated_file_names = run_yolov4(raw_files_dir, cfg, weights, coco_data, darknet_path, video)
                # video automatically put in the darknet directory. We will move it to a folder next to the raw video files. 
                copy_annotated_video(raw_files_dir, annotated_file_names, tl_folder_YOLO, darknet_path, args.weights, video_out)
        else:
            annotated_dir = return_annotated_directory(raw_files_dir, tl_folder_YOLO, args.weights)
            logevent(annotated_dir,1)
            run_yolov4(annotated_dir, cfg, weights, coco_data, darknet_path, video)
            logevent(f'annotated_dir = \t{annotated_dir}',1)

        logevent(f'darknet_folder=\t{darknet_path}',1)
        logevent(f'raw_files=\t{raw_files_dir}',1)
        logevent(f'weights=\t{weights}',1)
        logevent(f'cfg=\t{cfg}',1)
        logevent(f'data=\t{coco_data}',1)


    else:
        logevent("Running darknet command natively")
        if darknet_path:
            logevent(f"Executing command: cd {darknet_path} && {args.darknet_format}",1)
            exitcode = os.system(f'cd {darknet_path} && {args.darknet_format}')
            if exitcode != 0:
                logevent(f"Inference did not run successfully (exit code {exitcode})",3)
                        
        else:
            logevent(f"Executing command: {args.darknet_format}",1)
            exitcode = os.system(f'{args.darknet_format}')
            if exitcode != 0:
                logevent(f"Inference did not run successfully (exit code {exitcode})",3)


if __name__ == '__main__':
    main()
