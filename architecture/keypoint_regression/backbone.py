import tensorflow.keras as keras
from tensorflow.keras import layers


def vgg_model(IMG_SIZE, NUM_KEYPOINTS):
    vgg =  keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    vgg.trainable=False
    # x = Flatten()(vgg.output)

    inputs = layers.Input( (IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = vgg(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(
        NUM_KEYPOINTS, kernel_size=5, strides=1, activation="relu"
    )(x)
    outputs = layers.SeparableConv2D(NUM_KEYPOINTS, kernel_size=(3,3), strides=(2,2), activation='relu')(x)

    return keras.Model(inputs, outputs, name='keypoint_detector')



