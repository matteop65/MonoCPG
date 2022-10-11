# Table of Contents  
- [MonoCPG](#monocpg)  
- [Visualisation of Results](#visualisation-of-results)
- [Installation](#installation)
- [Running](#running)

# MonoCPG
MonoCPG is an infrastructure-based monocular 3D object detection model for autonomous driving. The model seeks to function as a novel 3D object detection tool for automated data annotation. Created as part of a 3rd year dissertation at the University of Warwick, in partnership with Intelligent Vehicles Research WMG. 

The current version of the model has been trained using a virtual environment ([CARLA](https://carla.org/)), providing a proof-of-concept. Thus, functions solely on trucks. Futher development is being conducted to generalise the model to various infrastructure camera positions and vehicle thypes.

For additional information, such as datasets, or model architecture, or performance metrics, please contact author: matteo.penlington@warwick.ac.uk 

# Visualisation of results
<img src="https://github.com/matteop65/MonoCPG/blob/main/result_image1.jpg" alt="SampleInputImage" width="500"/>
<img src="https://github.com/matteop65/MonoCPG/blob/main/result_image2.jpg" alt="SampleInputImage" width="500"/>
<img src="https://github.com/matteop65/MonoCPG/blob/main/result_image3.jpg" alt="SampleInputImage" width="500"/>


# Installation
Tested on Ubuntu 18.04 with python 3.7. Tested with gpu support using tensorflow 2.5.0. See [tensorflow-documentation](https://www.tensorflow.org/install/source#gpu) for all compatibility options.

The dependencies were installed in a conda environment. These can be found in environment.yml

To run with stock yolo, download the yolov4-p6.weights from [darknet](https://github.com/AlexeyAB/darknet/#pre-trained-models). Rename the file to yolov4-p6_best.weights. 

# Running

A folder containing all the output information is placed in *data/results*. If a *results* folder already exists, then it is renamed with the current UNIX time. 

| Argument | Description |
| --- | --- |
| `--dataset` | The path of the images in the dataset to run with. The images must be within the dataset folder. E.g., *data/datasetv1*. NB: Dataset folder must contain subfolder names "images" with images within it. </li></ul> |
| `--method` | The type of solving procedure you would like to use (i.e., 3, 4 or 5 keypoints) <ul><li>default: `5` </li></ul> |
| `--vgg_model_name` | The name of the trained model. The trained model must predict the same number of keypoints as stated in the method argument.|

