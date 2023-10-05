from sys import argv
from shutil import rmtree
import numpy as np
import cv2, os
from classifier_utils import *

#TODO: maybe use greyscale images for classification!

image_size = (64, 64)

def main(args):

    

    #assemble augmented data from example images, compile + train model
    if 'train' in args:

        labelsFile = open('../labels.txt', 'w+')

        images = []
        labels = []

        labelID = 1
        for imgFileName in os.listdir('../images'): # for each image in 'images' directory:
            imgClassName = imgFileName.split('.')[0] # get class name from image file name

            #ignore non-image files (file extension is not an image type)
            if imgFileName.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue

            #get greyscale cropped face from sample image
            (img, orig) = croppedFaceFromImg(cv2.imread('../images/' + imgFileName), image_size)
            if img is None: # ensure there is a face in the image
                print(f"{imgClassName} has no faces.")
                continue

            print(img.shape)

            print(f"new image class: {imgClassName}")
            labelsFile.write(f"{labelID} {imgClassName}\n") # add img class name to labels txt file
            

            images.append(img)
            labels.append(labelID)

            labelID += 1



        images = np.array(images)
        labels = np.array(labels)

        dataset = getAugmentedData(images, labels, image_size, 500)


        for imgs, labels in dataset:
            print(imgs[0].dtype, imgs[0].shape)

        model = keras.Sequential([
            keras.layers.Input(shape=image_size + (3,)),
            keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(len(labels), activation='softmax')  # Three units for three classes
        ])

        model.compile(loss='sparse_categorical_crossentropy',
            optimizer= keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        history = model.fit(
            dataset,
            epochs=3,
            callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
        )

        model.save('model.keras')

    if "run" in args:
        if not "model.keras" in os.listdir():
            print('"model.keras" model file not found')
            return
        
        labels = {}

        labelsFile = open('../labels.txt', 'r')

        for line in labelsFile.readlines():
            lineInfo = line.split(' ')
            labels[lineInfo[0]] = lineInfo[1][:lineInfo[1].index('\n')]

        model = keras.models.load_model('model.keras')
        
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        cv2.namedWindow('webcam')

        while True:
            success, frame = cam.read()
            if not success:
                return
            
            (croppedImg, orig) = croppedFaceFromImg(frame, image_size) 

            cv2.imshow('test', orig)
            cv2.waitKey(10)

            if croppedImg is None:
                continue
            

            croppedImg = (croppedImg / 255.0).reshape((1,) + croppedImg.shape) # normalize values to range of 0-1

            predictions = model.predict(croppedImg)

            # Get the predicted class index (assuming single class prediction)
            predicted_class_index = np.argmax(predictions)
            print(predicted_class_index, labels[str(predicted_class_index)])

main(argv)