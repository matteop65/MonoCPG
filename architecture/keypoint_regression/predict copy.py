from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
# from classes import KeyPointsDataset
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa

from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import cv2


def predict_keypoints(IMG_DIR, JSON):
    """
        Define hyperparameters
    """

    IMG_SIZE = 224
    BATCH_SIZE = 200
    EPOCHS = 10
    NUM_KEYPOINTS = 5 * 2  # 24 pairs each having x and y coordinates

    """
    Load data
    """

    # IMG_DIR = "Images"
    # JSON = "StanfordExtra_V12/StanfordExtra_v12.json"

    # # IMG_DIR = "datasetv1/images"
    # JSON='datasetv1.json'

    # Load the ground-truth annotations.
    with open(JSON) as infile:
        json_data = json.load(infile)

    # Set up a dictionary, mapping all the ground-truth information
    # with respect to the path of the image.

    # if error here bc no brackets at start of json
    json_dict = {i["img_path"]: i for i in json_data}


    # Utility for reading an image and for getting its annotations.
    def get_dog(name):
        data = json_dict[name]
        img_data = plt.imread(data["img_path"])
        # If the image is RGBA convert it to RGB.
        if img_data.shape[-1] == 4:
            img_data = img_data.astype(np.uint8)
            img_data = Image.fromarray(img_data)
            img_data = np.array(img_data.convert("RGB"))
        data["img_data"] = img_data

        return data


    """
    Visualize data
    """

    def visualize_keypoints(images, keypoints):
        fig, axes = plt.subplots(nrows=len(images), ncols=2, figsize=(16, 12))
        [ax.axis("off") for ax in np.ravel(axes)]

        for (ax_orig, ax_all), image, current_keypoint in zip(axes, images, keypoints):
            ax_orig.imshow(image)
            ax_all.imshow(image)

            # If the keypoints were formed by `imgaug` then the coordinates need
            # to be iterated differently.
            if isinstance(current_keypoint, KeypointsOnImage):
                for idx, kp in enumerate(current_keypoint.keypoints):
                    ax_all.scatter(
                        [kp.x], [kp.y], c='#ff0000', marker="x", s=50, linewidths=5
                    )
            else:
                current_keypoint = np.array(current_keypoint)
                # Since the last entry is the visibility flag, we discard it.
                current_keypoint = current_keypoint[:, :2]
                for idx, (x, y) in enumerate(current_keypoint):
                    ax_all.scatter([x], [y], c='#ff0000', marker="x", s=50, linewidths=5)

        plt.tight_layout(pad=2.0)
        plt.show()


    # Select four samples randomly for visualization.
    samples = list(json_dict.keys())
    # print(f'samples: {samples}')
    num_samples = 4
    selected_samples = np.random.choice(samples, num_samples, replace=False)

    images = []

    for sample in selected_samples:
        # print(sample)
        data = get_dog(sample)
        image = data["img_data"]
        # keypoint = data["keypoints"]

        images.append(image)
        # keypoints.append(keypoint)


    """
    Define augmentation transforms
    """

    train_aug = iaa.Sequential(
        [
            iaa.Resize(IMG_SIZE, interpolation="linear"),
            # iaa.Fliplr(0.3),
            # `Sometimes()` applies a function randomly to the inputs with
            # a given probability (0.3, in this case).
            # iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
        ]
    )

    test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])

    class KeyPointsDataset(keras.utils.Sequence):
        def __init__(self, image_keys, aug, batch_size=BATCH_SIZE, train=True):
            self.image_keys = image_keys
            self.aug = aug
            self.batch_size = batch_size
            self.train = train
            self.on_epoch_end()

        def __len__(self):
            return len(self.image_keys) // self.batch_size

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.image_keys))
            if self.train:
                np.random.shuffle(self.indexes)

        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
            image_keys_temp = [self.image_keys[k] for k in indexes]
            (images) = self.__data_generation(image_keys_temp)

            return (images)

        def __data_generation(self, image_keys_temp):
            batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
            # batch_keypoints = np.empty(
            #     (self.batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
            # )

            for i, key in enumerate(image_keys_temp):
                data = get_dog(key)
                # current_keypoint = np.array(data["keypoints"])[:, :2]
                # kps = []

                # To apply our data augmentation pipeline, we first need to
                # form Keypoint objects with the original coordinates.
                # for j in range(0, len(current_keypoint)):
                #     kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

                # We then project the original image and its keypoint coordinates.
                current_image = data["img_data"]
                # kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

                # Apply the augmentation pipeline.
                (new_image) = self.aug(image=current_image)
                batch_images[i,] = new_image

                # Parse the coordinates from the new keypoint object.
                # kp_temp = []
                # for keypoint in new_kps_obj:
                #     kp_temp.append(np.nan_to_num(keypoint.x))
                #     kp_temp.append(np.nan_to_num(keypoint.y))

                # More on why this reshaping later.
                # batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, NUM_KEYPOINTS)

            # # Scale the coordinates to [0, 1] range.
            # batch_keypoints = batch_keypoints / IMG_SIZE

            return (batch_images)


    """
    Create training and validation splits
    """

    # np.random.shuffle(samples)
    train_keys, validation_keys = (
        samples[int(len(samples) * 0.5) :],
        samples[: int(len(samples) * 0.5)],
    )
    inference_keys = samples

    # print(f'validation keys {os.path.split(validation_keys)[1]}')

    print(f'inference_keys: {inference_keys}')
    """
    Data generator investigation
    """

    train_dataset = KeyPointsDataset(train_keys, train_aug)
    validation_dataset = KeyPointsDataset(validation_keys, test_aug, train=False)
    inference_dataset = KeyPointsDataset(inference_keys, test_aug, train=False)

    # print(inference_dataset.im)

    """
    ## Make predictions and visualize them
    """

    model = keras.models.load_model('trained_models/testmodel1')

    sample_val_images = [get_dog(img)['img_data'] for img in inference_keys]
    # sample_val_images = next(iter(inference_dataset), None)
    print(f'sample_val_images: {sample_val_images}')
    print(f'length sample_val_images: {len(sample_val_images)}')
    # sample_val_images = sample_val_images[:10]
    # sample_val_keypoints = sample_val_keypoints[:4].reshape(-1, 5, 2) * IMG_SIZE
    predictions = model.predict(sample_val_images).reshape(-1, 5, 2) * IMG_SIZE
    # print(f'validation dataset.keys {validation_dataset.image_keys}')


    # print(predictions)
    # print(sample_val_images)

    # outputs
    for idx, num in enumerate(predictions):
        # output folder
        results_path = os.path.split(IMG_DIR)[0]
        img_name = f'{os.path.split(inference_dataset.image_keys[idx])[1]}'
        img_path = os.path.join(results_path, f'{os.path.splitext(img_name)[0]}.txt')

        with open(img_path, 'w') as f:
            for i, keypnt in enumerate(num): 
                u = keypnt[0] * 1920/224
                v = keypnt[1] * 1080/224
                print(f'ub: {keypnt[0]}, uf: {u}')
                f.write(str(u) +' '+ str(v) +'\n')
            # f.write('\n')

        # try:
        # image_file = Image.open(sample_val_images[idx])
        # img = image_file.convert('RGB')
        # img = Image.fromarray(img)
        # img.save(f'datasetv1/predicted_images/{img_name}.jpg')
        # plt.savefig()
            # cv2.imwrite(f'datasetv1/predicted_images/{img_name}.jpg',sample_val_images[idx])
        # except:
        #     print('failed to write image')
    # print(validation_dataset)

    # print(type(sample_val_images[idx]))
    # Ground-truth
    # visualize_keypoints(sample_val_images, sample_val_keypoints)

    # Predictions
    visualize_keypoints(sample_val_images, predictions)

