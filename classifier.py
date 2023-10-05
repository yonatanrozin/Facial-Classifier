"""
Facial recognition using cv2, keras and facenet
Yonatan Rozin
"""

from sys import argv

valid_args = ['train', 'run', 'reset', 'list']
msg = 'args must all be at least 1 of "train", "run", "reset" or "list".'
assert len(argv) > 1 and not False in [(arg in valid_args) for arg in argv[1:]], msg

import time
import cv2
import os
import numpy as np
from utils import *

from keras_facenet import FaceNet
facenet = FaceNet() # model to extract embeddings from cropped face images

if 'list' in argv:
    print([key for key in np.load('embeddings.npy', allow_pickle=True).item()])

if 'train' in argv:

    if 'reset' in argv:
        sample_embeddings = {} # dict to hold extracted embedding(s) per image class
    else:
        try:
            # load pre-trained data
            sample_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
        except:
            print("no .npy labels file found. Training from scratch.")
            sample_embeddings = {}
    
    for imgClass in os.listdir('images'):

        dirName = 'images/' + imgClass

        if not os.path.isdir(dirName):
            continue

        dirImages = os.listdir(dirName)
        if len(dirImages) == 0:
            print(f'empty directory: {dirName}')
            continue
        newClass = not imgClass in sample_embeddings
        newData = not newClass and len(dirImages) != len(sample_embeddings[imgClass])
        dataExists = not newClass and not newData

        if dataExists: 
            continue

        if newClass:
            print(f"new class: {imgClass}")
        elif newData:
            print(f"new data found for class: {imgClass}")

        class_embeddings = []

        for imgFile in os.listdir(dirName):
            
            imgFile = f"{dirName}/{imgFile}"
            img = cv2.imread(imgFile)
            
            # attempt extracting face from file (throws an error if file is not an image)
            try:
                faces = croppedFacesFromImg(img, 'MTCNN')
                if len(faces) == 0:
                    print(f'no face found in img {imgFile}')
                    continue
            except:
                print(f"error reading img {imgFile}")
                continue

            faceAreas = [w * h for (x, y, w, h) in faces]

            x, y, w, h = faces[faceAreas.index(max(faceAreas))]
            croppedFace = img[y:y+h, x:x+w]
            
            class_embeddings.append(facenet.embeddings([croppedFace]))

            #resize cropped face for preview + include label
            h_ratio = h/256
            previewImg = cv2.resize(croppedFace, (int(w/h_ratio), int(h/h_ratio)))
            cv2.putText(previewImg, imgClass, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50), 2)
            cv2.imshow('test', previewImg)
            cv2.waitKey(5)

        if len(class_embeddings) > 0:
            sample_embeddings[imgClass] = class_embeddings
            np.save('embeddings.npy', sample_embeddings)

    cv2.destroyAllWindows()

    

if "run" in argv:

    try:
        sample_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
    except:
        print('Model not trained. Run "classifier.py train" to train.')
        exit()

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    images = {}
    for imgClass in sample_embeddings.keys():
        dirName = "images/" + imgClass
        imgName = dirName + '/' + os.listdir(dirName)[0]
        images[imgClass] = cv2.imread(imgName)
            
    while True:

        success, frame = cam.read()

        if (success):

            #resize frame to 1000px high
            (frameH, frameW, _) = frame.shape
            frame_w_ratio = frameW/1000
            frame = cv2.resize(frame, (int(frameW/frame_w_ratio), int(frameH/frame_w_ratio)))

            output_frame = frame.copy()

            faces = croppedFacesFromImg(frame, 'haar')
            if len(faces) == 0:
                continue

            for i in range(len(faces)):

                (x, y, w, h) = faces[i]
                cropped = frame[y:y+h, x:x+w]

                cv2.rectangle(output_frame, (x, y), (x+w, y+h), 0, 3)

                # get top 5 matches
                matches = classifyFace(cropped, sample_embeddings)[:5]

                h_ratio = h/100
                cropped = cv2.resize(cropped, (int(w/h_ratio), int(h/h_ratio)))
                (faceH, faceW, _) = cropped.shape

                output_frame[i * 200 + 20 : i * 200 + 20 + faceH, 20 : 20 + faceW] = cropped

                matchX = 200

                for j in range(len(matches)):
                    match_class = matches[j][0]
                    match_confidence = int((1-matches[j][1]) * 100)

                    if match_confidence < 10:
                        continue

                    match = images[match_class]
                    (h, w, _) = match.shape
                    h_ratio = h/100
                    match = cv2.resize(match, (int(w/h_ratio), int(h/h_ratio)))
                    
                    (h, w, _) = match.shape # new dimensions after resizing
                    output_frame[i * 200 + 20 : i * 200 + 20 + h, matchX : matchX + w] = match

                    cv2.putText(output_frame, match_class, (matchX, i*200+20 + h + 20), cv2.FONT_HERSHEY_COMPLEX, .7, (255,0,0))
                    cv2.putText(output_frame, str(match_confidence)+"%", (matchX, i*200+20 + h + 50), cv2.FONT_HERSHEY_COMPLEX, .7, (255,0,0))

                    matchX += 150
                    
            cv2.imshow('test', output_frame)
            cv2.waitKey(5)
