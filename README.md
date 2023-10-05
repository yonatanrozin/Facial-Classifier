# Facial Classifier
A python script that uses a trained neural network to recognize faces from the webcam

## Dependencies + Installation
- Dependencies
  - [Python](https://www.python.org/) - tested with version 3.11.6
  - Python packages (use ```pip``` to install)
    - [Numpy](https://pypi.org/project/numpy/) - tested with version 1.24.3
    - [OpenCV-Python](https://pypi.org/project/opencv-python/) - tested with version 4.8.1
    - [Keras-FaceNet](https://pypi.org/project/keras-facenet/) - tested with version 0.3.2
    - [MTCNN](https://pypi.org/project/mtcnn/) - tested with version 0.1.1
- ```git clone``` this repository into desired location on local hard drive

## Training
- ```cd``` into cloned repo folder
- create image class sub-directories inside ```images``` directory. The name of the sub-directory will be used as the name of the class
  - insert any number of sample images into sub-directory, though the model already works well with only 1!
- run ```python3 classifier.py train``` to begin training. May take a while depending on number of provided image classes
  - run this command any time new data is added
  - Or run ```python3 classifier.py train reset``` to reset and train all image classes from scratch (required if removing a class)

## Running
- ```cd``` into cloned repo folder
- run ```python3 classifier.py run``` to begin running. A new window with the webcam view and classification results should pop up automatically.
- press Q at any time to quit
