# from gc import callbacks
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow as tf
from training.backbone_architectures import vgg_model
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

def training_model(json_data, model_name, IMG_SIZE, BATCH_SIZE, NUM_KEYPOINTS, EPOCHS):
    """
        Define hyperparameters
    """

    # IMG_SIZE = 224
    # BATCH_SIZE = 64
    # EPOCHS = 10
    # NUM_KEYPOINTS = 5 * 2  # 24 pairs each having x and y coordinates
    BATCH_NUMBER = 1000

    """
    Load data
    """

    # IMG_DIR = "Images"
    # JSON = "StanfordExtra_V12/StanfordExtra_v12.json"

    # IMG_DIR = "datasetv1/images"
    # JSON='datasetv1.json'

    # Load the ground-truth annotations.
    # with open(JSON) as infile:
    #     json_data = json.load(infile)

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
    num_samples = 4
    selected_samples = np.random.choice(samples, num_samples, replace=False)

    images, keypoints = [], []

    for sample in selected_samples:
        print(sample)
        data = get_dog(sample)
        image = data["img_data"]
        keypoint = data["keypoints"]

        images.append(image)
        keypoints.append(keypoint)

    # print(f'images: {images}')
    # print(f'keypoints: {keypoints}')
    visualize_keypoints(images, keypoints)

    """
    Define augmentation transforms
    """

    train_aug = iaa.Sequential(
        [
            iaa.Resize(IMG_SIZE, interpolation="linear"),
            iaa.Fliplr(0.3),
            # `Sometimes()` applies a function randomly to the inputs with
            # a given probability (0.3, in this case).
            iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7))),
        ]
    )

    test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation="linear")])

    class KeyPointsDataset(keras.utils.Sequence):
        def __init__(self, image_keys, aug, batch_size=BATCH_SIZE, batch_number=BATCH_NUMBER, train=True):
            self.image_keys = image_keys
            self.aug = aug
            self.batch_size = batch_size
            self.train = train
            self.batch_number = batch_number
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
            images_total = []
            keypoints_total = []
            # for i in range(self.batch_number):
            (images, keypoints) = self.__data_generation(image_keys_temp)
                # images_total.append(images)
                # keypoints_total.append(keypoints)

            # print(f'len images_total: {len(images_total)}')
            return (images, keypoints)

        def __data_generation(self, image_keys_temp):
            batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype="int")
            batch_keypoints = np.empty(
                (self.batch_size, 1, 1, NUM_KEYPOINTS*2), dtype="float32"
            )

            for i, key in enumerate(image_keys_temp):
                data = get_dog(key)
                current_keypoint = np.array(data["keypoints"])[:, :2]
                kps = []

                # To apply our data augmentation pipeline, we first need to
                # form Keypoint objects with the original coordinates.
                for j in range(0, len(current_keypoint)):
                    kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))

                # We then project the original image and its keypoint coordinates.
                current_image = data["img_data"]
                kps_obj = KeypointsOnImage(kps, shape=current_image.shape)

                # Apply the augmentation pipeline.
                (new_image, new_kps_obj) = self.aug(image=current_image, keypoints=kps_obj)
                batch_images[i,] = new_image

                # Parse the coordinates from the new keypoint object.
                kp_temp = []
                for keypoint in new_kps_obj:
                    kp_temp.append(np.nan_to_num(keypoint.x))
                    kp_temp.append(np.nan_to_num(keypoint.y))

                # More on why this reshaping later.
                batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, NUM_KEYPOINTS*2)

            # Scale the coordinates to [0, 1] range.
            batch_keypoints = batch_keypoints / IMG_SIZE

            return (batch_images, batch_keypoints)


    """
    Create training and validation splits
    """
    # samples = np.concatenate(samples, samples, samples)
    for j in range(BATCH_NUMBER):
        for i in range(200):
            # print(i)
            samples.append(samples[i])
    print(f'len samples: {len(samples)}')

    # for idx, num in enumerate(samples):
    #     # print(idx)
    #     samples.append(num)


    np.random.shuffle(samples)
    # print(f'samples: {samples}')
    # print(int(len(samples)*0.15))
    # print(samples[30:])
    train_keys, validation_keys = (
        samples[int(len(samples) * 0.6) :],
        samples[: int(len(samples) * 0.4)],
    )

    # print(f'train_keys: {train_keys}')

    # train_keys, validation_keys = (
    #     samples[3],
    #     samples[1],
    # )


    """
    Data generator investigation
    """

    train_dataset = KeyPointsDataset(train_keys,train_aug, batch_number=EPOCHS, train=True)
    validation_dataset = KeyPointsDataset(validation_keys, test_aug, batch_number=EPOCHS, train=False)


    print(f"Total batches in training set: {len(train_dataset)}")
    print(f"Total batches in validation set: {len(validation_dataset)}")

    sample_images, sample_keypoints = next(iter(train_dataset))
    # print(sample_keypoints)
    # assert sample_keypoints.max() == 1.0
    # assert sample_keypoints.min() == 0.0

    sample_keypoints = sample_keypoints[:4].reshape(-1, NUM_KEYPOINTS, 2) * IMG_SIZE
    # print(sample_images[:4], sample_keypoints)
    visualize_keypoints(sample_images[:4], sample_keypoints)


    """
        Model Building
    """

    # def vgg_model():
    #     vgg =  keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    #     vgg.trainable=False
    #     # x = Flatten()(vgg.output)

    #     inputs = layers.Input( (IMG_SIZE, IMG_SIZE, 3))
    #     x = keras.applications.vgg16.preprocess_input(inputs)
    #     x = vgg(x)
    #     x = layers.Dropout(0.3)(x)
    #     x = layers.SeparableConv2D(
    #         NUM_KEYPOINTS*2, kernel_size=5, strides=1, activation="relu"
    #     )(x)
    #     outputs = layers.SeparableConv2D(NUM_KEYPOINTS*2, kernel_size=(3,3), strides=(2,2), activation='relu')(x)

    #     return keras.Model(inputs, outputs, name='keypoint_detector')


    model = vgg_model(IMG_SIZE, NUM_KEYPOINTS*2)
    model.summary()
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-4))
    checkpoint = ModelCheckpoint("trained_models/best_training_model", monitor="loss", verbose=1, save_best_only=True, mode='auto', save_freq=64)
    model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS, callbacks=[checkpoint])

    save_model_path = f'trained_models/{model_name}'
    model.save(save_model_path)

    print(f'saved model too: {save_model_path}')




















































