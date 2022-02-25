import json
import cv2
import os
from architecture.keypoint_regression.predict_o import predict_keypoints
from architecture.keypoint_regression.write_to_json import create_json

def run_regression(results_path, raw_images_path, cropped_path):
    """
        runs keypoint regression given cropped images, 
            - creates json
            - gets predictions
            - scales keypoints to original img resolution
    """


    """
        create json
    """
    json_pth = os.path.join(results_path, 'annotations.json')
    relative_images_path = create_json(cropped_path, json_pth)


    """
        get predictions
    """
    with open(json_pth) as infile:
        json_data = json.load(infile)
    
    predictions, inference_dataset = predict_keypoints(json_data)


    """
        scale keypoints to original img resolution
    """
    img_width =[i['img_width'] for i in json_data]
    img_height =[i['img_height'] for i in json_data]

    scaled_keypnts = []
    for idx, num in enumerate(predictions):
        img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'

        uv = []
        for i, keypnt in enumerate(num):
            u = keypnt[0] * img_width[i] / 224
            v = keypnt[1] * img_height[i] / 224
            uv.append( [u, v])

        scaled_keypnts.append(uv)


    """
        Annotate cropped images with predicted keypnts
    """
    for idx, num in enumerate(scaled_keypnts):
        img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'
        cropped_img_path = os.path.join(cropped_path, img_name)
        cropped_img = cv2.imread(cropped_img_path)

        colours = [ [0,0,255], [255, 255, 0], [0,255,0], [0,255,255], [0,0,0]]
        # annotate img
        annotated_img = cropped_img
        for i, keypnt in enumerate(num):
            annotated_img = cv2.circle(annotated_img, (int(keypnt[0]), int(keypnt[1])), 4, colours[i], -1)
        
        # create cropped img predictions directory
        cropped_img_pred_fol = os.path.join(results_path, 'cropped_img_predictions')
        if not os.path.isdir(cropped_img_pred_fol):
            os.system(f'mkdir {cropped_img_pred_fol}')

        # save new image
        new_img_path = os.path.join(cropped_img_pred_fol, img_name)
        cv2.imwrite(new_img_path, annotated_img)


    """
        transpose them to their original position
    """
    predicted_keypnts_folder = os.path.join(results_path, 'predicted_keypnts_local')

    if not os.path.isdir(predicted_keypnts_folder):
        os.system(f'mkdir {predicted_keypnts_folder}')

    for idx, num in enumerate(scaled_keypnts):
        img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'
        img_path = f'{predicted_keypnts_folder}/{os.path.splitext(img_name)[0]}.txt'
        with open(img_path, 'w') as f:
            for i, keypnt in enumerate(num): 
                u = keypnt[0]
                v = keypnt[1]
                print(f'ub: {keypnt[0]}, uf: {u}')
                f.write(str(u) +' '+ str(v) +'\n')
    
    return predicted_keypnts_folder, colours