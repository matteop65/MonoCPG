import json
import cv2
import os
from architecture.keypoint_regression.predict_o import predict_keypoints
from architecture.keypoint_regression.write_to_json import create_json





def run_regression(results_path, raw_images_path, cropped_path, model_name, num_keypoints, v):
    """
        runs keypoint regression given cropped images, 
            - creates json
            - gets predictions
            - scales keypoints to original img resolution
    """
    # print(f'\n\n\n\n')


    """
        create json
    """
    json_pth = os.path.join(results_path, 'annotations.json')
    relative_images_path = create_json(cropped_path, json_pth, train=False, method=0)


    """
        get predictions
    """
    with open(json_pth) as infile:
        json_data = json.load(infile)
    
    predictions, inference_dataset, sample_val_images = predict_keypoints(json_data, model_name, num_keypoints,visualise=v)
    # print(f'predictions: {predictions}')

    """ this was a temporary output to see what the sample_val_images looked liked"""
    # # images input to the model
    # img_names = [i['img_name'] for i in json_data]
    # for idx, img in enumerate(sample_val_images):
    #     name = img_names[idx]
    #     path= os.path.join(results_path, 'predict_output_images')
    #     if not os.path.isdir(path):
    #         os.system(f'mkdir {path}')

    #     new_img_pth = os.path.join(path,name )
    #     cv2.imwrite(new_img_pth, img)


    # for idx, num in enumerate(predictions):
    #     # img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'
    #     name =img_names[idx]
    #     # print(f'type(image_name: {type(image_name)}')
    #     # print(f'image_name: {image_name}')
    #     path= os.path.join(results_path, 'predict_output_images')
    #     if not os.path.isdir(path):
    #         os.system(f'mkdir {path}')
        
    #     new_img_pth = os.path.join(path,name )
    #     cv2.imwrite(new_img_pth, sample_val_images[idx])

    #     cropped_img = cv2.imread(new_img_pth)

    #     colours = [ [0,0,255], [255, 255, 0], [0,255,0], [0,255,255], [0,0,0]]
    #     # annotate img
    #     annotated_img = cropped_img
    #     for i, keypnt in enumerate(num):
    #         annotated_img = cv2.circle(annotated_img, (int(keypnt[0]), int(keypnt[1])), 4, colours[i], -1)
        
    #     # create cropped img predictions directory
    #     # cropped_img_pred_fol = os.path.join(results_path, 'cropped_img_predictions')
    #     # if not os.path.isdir(cropped_img_pred_fol):
    #     #     os.system(f'mkdir {cropped_img_pred_fol}')

    #     # save new image
    #     # new_img_path = os.path.join(cropped_img_pred_fol, name)
    #     cv2.imwrite(new_img_pth, annotated_img)


    """
        scale keypoints to original img resolution
    """
    img_width =[i['img_width'] for i in json_data]
    img_height =[i['img_height'] for i in json_data]
    img_name = [i['img_name'] for i in json_data]



    scaled_keypnts = []
    for idx, num in enumerate(predictions):
        # img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'

        uv = []
        for i, keypnt in enumerate(num):
            # print(f'image_name: {img_name[idx]}')
            # print(f'imagesize: {img_width[idx], img_height[idx]}')
            # print(f'old keypnt: {keypnt[0], keypnt[1]}')
            u = keypnt[0] * img_width[idx] / 224
            v = keypnt[1] * img_height[idx] / 224
            # print(f'new keypnt: {u,v}')
            uv.append( [u, v])

        scaled_keypnts.append(uv)


    """
        Annotate cropped images with predicted keypnts
    """
    for idx, num in enumerate(scaled_keypnts):
        # img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'
        image_name =str(img_name[idx])
        # print(f'type(image_name: {type(image_name)}')
        # print(f'image_name: {image_name}')
        cropped_img_path = os.path.join(cropped_path, image_name)
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
        new_img_path = os.path.join(cropped_img_pred_fol, image_name)
        cv2.imwrite(new_img_path, annotated_img)


    """
        save local keypnts
    """
    predicted_keypnts_folder = os.path.join(results_path, 'predicted_keypnts_local')
    output_keypnts = os.path.join(results_path, 'output_keypnts')
    if not os.path.isdir(predicted_keypnts_folder):
        os.system(f'mkdir {predicted_keypnts_folder}')
    if not os.path.isdir(output_keypnts):
        os.system(f'mkdir {output_keypnts}')

    # scaled outputs
    for idx, num in enumerate(scaled_keypnts):
        txt_path = f'{predicted_keypnts_folder}/{os.path.splitext(img_name[idx])[0]}.txt'
        with open(txt_path, 'w') as f:
            for i, keypnt in enumerate(num): 
                u = keypnt[0]
                v = keypnt[1]
                # print(f'ub: {keypnt[0]}, uf: {u}')
                f.write(str(u) +' '+ str(v) +'\n')

    # original outputs
    for idx, num in enumerate(predictions):
        l_txt_path = f'{output_keypnts}/{os.path.splitext(img_name[idx])[0]}.txt'
        with open(l_txt_path,'w') as f:
            for i, keypnt in enumerate(num):
                u = keypnt[0]
                v = keypnt[1]
                f.write(f'{str(u)} {str(v)}\n')
    
    # print(f'\n\n\n\n')
    return predicted_keypnts_folder, colours