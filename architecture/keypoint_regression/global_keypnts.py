import cv2
from logfile import logevent

import os

def transform_to_global_keypnts(bbox_path, local_keypnts_path, img_path, colours):
    """
        Given bbox and keypoints of an image, find global keypnts
    """
    img = cv2.imread(img_path)

    # open txt file
    with open(bbox_path, 'r') as bbxfile:
        bbxcontents = bbxfile.readlines()

    # find top left of 2d bbox (with increased size)
    if len(bbxcontents) == 1:
        txtcontents = bbxcontents[0]
        # print(f'txtcontents: {txtcontents}')

        content = txtcontents.split()
        bbx_u = int(content[0])
        bbx_v = int(content[1])

        # print(f'bbox u,v: {bbx_u, bbx_v}')
    else:
        logevent('no functionality for images with multiple detections', 4)

    with open(local_keypnts_path, 'r') as keypnts_file:
        keypnts_contents = keypnts_file.readlines()

    # print(f'keypnts_contents: {keypnts_contents}')
    # keypnts_contents.splitext(" ")
    # transpose local to global coordinates
    trsp_coords = []
    for idx, num in enumerate(keypnts_contents):
        num = num[:-1]
        # print(f'num: {num}')
        # print(f'type(num): {type(num)}')
        num = num.split(" ")
        # print(f'num: {num}')
        # print(f'num[0]: {float(num[0])}')
        # print(f'num[1]: {float(num[1])}')
        transpose_u = float(num[0]) + bbx_u
        transpose_v = float(num[1]) + bbx_v
        trsp_coords.append([transpose_u, transpose_v])

        # draw keypnts onto image
        img = cv2.circle(img, (int(transpose_u), int(transpose_v)), 3, colours[idx], -1)


    # save new txt file
    results_path = os.path.split(os.path.split(bbox_path)[0])[0]
    global_keypnts_fldr = os.path.join(results_path, 'global_keypnts')
    if not os.path.isdir(global_keypnts_fldr):
        os.system(f'mkdir {global_keypnts_fldr}')

    # create file
    txt_name = os.path.split(local_keypnts_path)[1]
    txt_path = os.path.join(global_keypnts_fldr, txt_name)
    # print(f'txt_name: {txt_name}')
    # print(f'txt path: {txt_path}')
    for idx, keypnt, in enumerate(trsp_coords):
        # txt_path = os.path.join(global_keypnts_fldr, )
        with open(txt_path, 'a+') as f:
            f.write(f'{int(keypnt[0])} {int(keypnt[1])}\n')

    
    # save img
    img_name = os.path.splitext(txt_name)[0] + '.jpg'
    img_path = os.path.join(global_keypnts_fldr, img_name)
    cv2.imwrite(img_path, img)