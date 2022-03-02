import json
import os
from PIL import Image

"""
    Look at VGG Keypoint repo for write_to_json for training
"""

def test():
    dict = {
        "img_path":"016501.jpg",
        "img_width":1920,
        "img_height": 1080,
        "keypoints":[ [826, 569], [1251, 759], [1324, 682], [1275,616], [884,636]]
    }



    json_object = json.dumps(dict, indent=4)
    with open('sample.json', 'w') as f:
        f.write(json_object)


def json_info(json_path, img_path, img_name, img_width, img_height, keypoints, end):
    dictionary = {
        "img_path":img_path, 
        "img_name":img_name,
        "img_width":img_width,
        "img_height":img_height,
        "keypoints":keypoints
    }

    json_object = json.dumps(dictionary, indent=4)
    write_to_json(json_path, 'a+', json_object)

    if end != 1:
        write_to_json(json_path, 'a+', ',')


def write_to_json(json_path, v, msg):
    """
        Will write or append to json at json_path
        json_path is path of json
        v is value either w or a
    """
    with open(json_path, v) as f:
        f.write(msg)



def create_json(images_path, json_path, method, train):

    # json path
    # json_path = 'datasetv1.json'
    if os.path.isfile(json_path):
        decision = input(f"json already exists as {json_path}. To overwrite [y]yes, [n]no:")
        if decision=='n':
            raise(Exception(f"json already exists! {json_path}"))
        elif decision=='y':
            write_to_json(json_path, 'w', '[')
            pass
        else:
            raise(Exception(f'Passed non acceptable argument. Try again.'))
    else:
        write_to_json(json_path, 'a', '[')


    cnt = 0
    # find the number of images. If the last image then a comma should not be added onto json
    for dirpath, dirname, filenames in os.walk(images_path):
        for filename in filenames:
            cnt += 1

    cropped_folder = os.path.split(images_path)[1]


    relative_images_path = []
    number=0
    # get information for json in relevant folder
    for dirpath, dirname, filenames in os.walk(images_path):
        for idx, filename in enumerate(filenames):
            # this is only for mac, sometimes .DS_Store is present in folder
            if filename==".DS_Store":
                raise(Exception("remove .DS_Store"))

            # relative file paths
            if train==False:
                img_path = f'data/results/{cropped_folder}/{filename}'
            elif train==True:
                img_path = f'{images_path}/{filename}'
            relative_images_path.append(img_path)

            img_name = os.path.split(img_path)[1]
            # get image dimensions
            im = Image.open(img_path)
            width, height = im.size

            if train==False:
                keypoints = [ [0,0], [0,0], [0,0], [0,0], [0,0]]
            elif train==True:
                keypoint_path = os.path.join(os.path.split(images_path)[0], 'gt_keypnts',os.path.splitext(filename)[0]+'.txt')

                # get contents of .txt files
                with open(keypoint_path, "r") as f:
                    contents = f.readlines()

                # remove keypnts if method==4,3
                if method == 4:
                    print(f'removed: {contents[:-1]}')
                    contents = contents[:-1]
                elif method == 3:
                    contents = contents[:-2]
                
                keypoints = []
                for idx, num in enumerate(contents):
                    # get each keypoint in .txt
                    temp_keypoint = num.split()

                    # convert keypoints from str to int
                    temp_keypoint = [int(i) for i in temp_keypoint]
                    
                    # append to keypoints file
                    keypoints.append(temp_keypoint)
            number +=1
            # print(f'idx: {number} out of {cnt}')
            if number == cnt:
                json_info(json_path, img_path, img_name, width, height, keypoints, end=1)
            else:
                json_info(json_path, img_path, img_name, width, height, keypoints, end=0)

    write_to_json(json_path, 'a', ']')

    return relative_images_path


if __name__ == '__main__':
    # test()
    # main()
    pass