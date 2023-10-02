import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.layers
import numpy as np

from tensorflow import data

# input: a cv2.Mat, probably loaded from an image file with imread or a video frame with VideoRecorder.read
def croppedFaceFromImg(img, size):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgGreyRGB = cv2.cvtColor(imgGrey, cv2.COLOR_GRAY2BGR)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faceData = face_classifier.detectMultiScale(imgGrey, 1.3, 5)
    
    if len(faceData) == 0: # no faces in image - return null
        return None
    elif len(faceData > 1):  # multiple faces - get largest one
        parallel_list = [w * h for x, y, w, h in faceData]
        faceData = faceData[parallel_list.index(max(parallel_list))]
    else: # one face - use it
        faceData = faceData[0]

    x, y, w, h = faceData

    return  cv2.resize(imgGreyRGB[y:y+h, x:x+w], size)

# inputs:   images/labels - parallel numpy arrays containing sample images + label ints 
#           numCopies - number of augmented copies to create PER sample
def getAugmentedData(images, labels, imgSize, numCopies = 100 ):
    print(images.shape, labels.shape)

    augmented_images = []
    augmented_labels = []

    data_augmentation_preprocessor = Sequential([
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomFlip("horizontal"),
        # keras.layers.RandomTranslation(.3, .3,  "nearest"),
        keras.layers.RandomZoom((-.5,0)),
        keras.layers.RandomBrightness(.2),
        keras.layers.RandomContrast(.5),
        keras.layers.Resizing(imgSize[0], imgSize[1]),
        keras.layers.Rescaling(1.0/255.0)
    ])


    for img, label in zip(images, labels):
        for i in range(numCopies):
            augmented_image = data_augmentation_preprocessor(img)
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
            # cv2.imshow('test', augmented_image.numpy())
            # cv2.waitKey(1)

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    training_data = data.Dataset.from_tensor_slices((augmented_images, augmented_labels))
    training_data = training_data.shuffle(buffer_size=len(augmented_images)) 
    training_data = training_data.batch(16)

    return training_data