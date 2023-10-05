import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

face_detector = MTCNN()

face_detector_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

facenet = FaceNet()

# inputs: 
#   img: a cv2.Mat, probably generated from cv2.imshow or cv2.imread
#   API: either "MTCNN" or "haar" - haar works MUCH faster but only works on frontal images
#   imgSize : (default 256x256) - a tuplet of sizes for images to be converted to (necessary?)
# returns:
#   a tuplet containing (original image with faces outlined, a list of cropped faces)
def croppedFacesFromImg(img, API):

    assert API in ["haar", "MTCNN"], "'API' arg should be one of 'haar' or 'MTCNN'."

    faceBoxes = []

    if API == "MTCNN":
        face_data = face_detector.detect_faces(img)
        for face in face_data:
            x, y, w, h = face['box']
            faceBoxes.append((x, y, w, h))

    elif API == "haar":
        imgGrey= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceData = face_detector_haar.detectMultiScale(imgGrey, 1.3, 3)
        for (x, y, w, h) in faceData:
            faceBoxes.append((x, y, w, h))
    
    return faceBoxes

def classifyFace(faceImg, sample_embeddings):
    embeddings = facenet.embeddings([faceImg])
    
    predictions = {}
    for class_name in sample_embeddings:
        dist_total = 0
        sample_img_count = len(sample_embeddings[class_name])
        for emb in sample_embeddings[class_name]: 
            dist_total += facenet.compute_distance(emb[0], embeddings[0])

        predictions[class_name] = min(max(0, dist_total / sample_img_count), 1)

    predictions_sorted = sorted(list(predictions.items()), key=lambda x: x[1])

    return predictions_sorted
