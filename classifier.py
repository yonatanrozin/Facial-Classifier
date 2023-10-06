"""
Facial recognition using cv2, keras and facenet
Yonatan Rozin
"""

from sys import argv
import time

valid_args = ['train', 'run', 'reset', 'list', 'debug']
msg = f'args must all be at least 1 of {valid_args}.'
assert len(argv) > 1 and not False in [(arg in valid_args) for arg in argv[1:]], msg

if 'list' in argv:
    import numpy as np
    try:
        print([key for key in np.load('embeddings.npy', allow_pickle=True).item()])
    except:
        print("Error finding embeddings.npy. Run 'python3 classifier.py train' first to assemble facial embeddings.")

if 'train' in argv:

    print('importing modules...')

    train_start_time = time.time()

    import cv2, os
    from keras_facenet import FaceNet
    from utils import *
    import numpy as np
    facenet = FaceNet() # model to extract embeddings from cropped face images

    if 'reset' in argv:
        sample_embeddings = {} # dict to hold extracted embedding(s) per image class
    else:
        try:
            # load pre-trained data
            sample_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
        except:
            print("no .npy labels file found. Training from scratch.")
            sample_embeddings = {}
    
    # iterate through sub-directories in 'images', using dir name as class name

    print('fetching sample images.')
    for imgClass in os.listdir('images'):

        dirName = 'images/' + imgClass
        print(f'directory: {dirName}')

        if not os.path.isdir(dirName):
            continue

        dirImages = os.listdir(dirName)
        if len(dirImages) == 0: #ignore empty directories
            print(f'empty directory: {dirName} - skipping.')
            continue
        newClass = not imgClass in sample_embeddings
        newData = not newClass and len(dirImages) != len(sample_embeddings[imgClass])
        dataExists = not newClass and not newData

        if dataExists: #ignore data that has already been added
            print(f"data already exists for class {imgClass}")
            continue
        if newClass:
            print(f"new class: {imgClass}")
        elif newData:
            print(f"new data found for class: {imgClass}")

        class_embeddings = [] #array to hold face embeddings for current class

        #per image inside sub-directory - get facial embeddings from cropped face
        for imgFile in os.listdir(dirName): 
            imgFile = f"{dirName}/{imgFile}"
            print(f'image: {imgFile}')
            img = cv2.imread(imgFile)

            
            try:
                print(f"extracting face...")
                faces = croppedFacesFromImg(img, 'MTCNN') #use MTCNN (not haar) for accuracy
                if len(faces) == 0:
                    print(f'no face found in img {imgFile}')
                    continue
            except Exception as e:
                print(e)
                print(f"error reading img {imgFile}")
                continue

            # only consider largest face per image
            faceAreas = [w * h for (x, y, w, h) in faces]
            x, y, w, h = faces[faceAreas.index(max(faceAreas))]
            croppedFace = img[y:y+h, x:x+w]
            
            class_embeddings.append(facenet.embeddings([croppedFace]))

            #display normalized height image + label while training
            h_ratio = h/256
            previewImg = cv2.resize(croppedFace, (int(w/h_ratio), int(h/h_ratio)))
            cv2.putText(previewImg, imgClass, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50), 2)
            cv2.imshow('test', previewImg)
            cv2.waitKey(5)

        #add facial embeddings for this class to dictionary + save to file
        if len(class_embeddings) > 0:
            sample_embeddings[imgClass] = class_embeddings
            print("Facial embeddings retrieved, saving to file.")
            np.save('embeddings.npy', sample_embeddings)

        print()

    cv2.destroyAllWindows()

    print(f"Classifier with {len(sample_embeddings)} classes trained in {(time.time() - train_start_time):.2f}s.")

if "run" in argv:

    print("importing modules...")

    import numpy as np
    import cv2, os
    from utils import *

    image_margin = 20 # space between displayed images and window border
    matches_offset_initial = 200 # horizontal starting position of displayed matched images
    matches_offset_increment = 150 # increment of horizontal position per matched image
    matched_image_height = 100 # normalized height of all displayed images
    matches_row_height = 200 # total row height per matched face (includes labels)

    try:
        print('loading facial embeddings file...')
        sample_embeddings = np.load('embeddings.npy', allow_pickle=True).item()
    except:
        print('Error loading file, or embeddings not yet obtained. Run "classifier.py train" to retrieve embeddings.')
        exit()

    cam_id = 0 # change this to desired camera ID (depending on # of available cameras)

    print("***Opening webcam for video capture. This line of code may differ between devices!!\n" + 
              "See https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#ad890d4783ff81f53036380bd89dd31aa for more info.")
    
    # cam = cv2.VideoCapture(cam_id)
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

    # get match images ahead of time to avoid frequent file system reading
    print('retrieving sample images per class...')
    images = {}
    for imgClass in sample_embeddings.keys():
        dirName = "images/" + imgClass
        imgName = dirName + '/' + os.listdir(dirName)[0]
        images[imgClass] = cv2.imread(imgName)

    print()
            
    while True:

        print('retrieving new frame.')
        success, frame = cam.read()

        if (success):

            #resize frame to 1000px wide, maintaining proportions
            (frameH, frameW, _) = frame.shape
            frame_w_ratio = frameW/1000
            frame = cv2.resize(frame, (int(frameW/frame_w_ratio), int(frameH/frame_w_ratio)))

            output_frame = frame.copy()

            #get cropped face
            print('extracting face(s) from frame.')
            faces = croppedFacesFromImg(frame, 'haar')
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

            #per matched face from webcam
            print('getting top 5 matches.')
            for i in range(len(faces)):

                #get cropped face
                (x, y, w, h) = faces[i]
                cropped = frame[y:y+h, x:x+w]

                matches = classifyFace(cropped, sample_embeddings)[:5] # get 5 strongest matches

                if min([match[1] for match in matches]) > .6:
                    print('no strong match found. skipping this frame.')
                    continue

                print('displaying scanned face(s)')
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), 0, 3)

                # normalize all image heights
                h_ratio = h/matched_image_height
                cropped = cv2.resize(cropped, (int(w/h_ratio), int(h/h_ratio)))

                # display cropped face
                (faceH, faceW, _) = cropped.shape
                output_frame[i * matches_row_height + image_margin : i * matches_row_height + image_margin + faceH, image_margin : image_margin + faceW] = cropped

                matches_offset = matches_offset_initial


                #for each match, display normalized height sample image, class label + confidence score 
                print('displaying matches.')
                for j in range(len(matches)):
                    match_class = matches[j][0]
                    match_confidence = int((1-matches[j][1]) * 100)

                    # normalize image height + overlay on webcam frame
                    match = images[match_class]
                    (h, w, _) = match.shape
                    h_ratio = h/matched_image_height
                    match = cv2.resize(match, (int(w/h_ratio), int(h/h_ratio)))
                    
                    (h, w, _) = match.shape
                    output_frame[i * matches_row_height + image_margin : i * matches_row_height + image_margin + h, matches_offset : matches_offset + w] = match

                    # show labels: class and confidence
                    cv2.putText(output_frame, match_class, (matches_offset, i*matches_row_height+image_margin + h + image_margin), cv2.FONT_HERSHEY_COMPLEX, .7, (255,0,0))
                    cv2.putText(output_frame, str(match_confidence)+"%", (matches_offset, i*matches_row_height+image_margin + h + 50), cv2.FONT_HERSHEY_COMPLEX, .7, (255,0,0))

                    matches_offset += matches_offset_increment

            print()
                    
            cv2.imshow('Classifier', output_frame)
            if cv2.waitKey(5) == ord('q'):
                exit()

    