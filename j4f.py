import numpy as np
import cv2
from tensorflow.keras.models import load_model
face_cascade = cv2.CascadeClassifier('C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\haarcascade_eye.xml')
fear = cv2.imread("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\fear.jpg")
fear = cv2.resize(fear,(167,167))
angry =cv2.imread("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\angry.png")
angry = cv2.resize(angry,(167,167))
disgust= cv2.imread("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\disgust.jpg")
disgust=cv2.resize(disgust,(167,167))
sad= cv2.imread("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\sad.png")
sad=cv2.resize(sad,(167,167))
surprise =cv2.imread("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\surprise.png")
surprise = cv2.resize(surprise,(167,167))
video_capture = cv2.VideoCapture(0)
emotion_dict= {0:angry, 5:sad, 'Neutral': 4, 1: disgust, 6:surprise, 2:fear, 'Happy': 3}
model = load_model("C:\\Users\\dupst\\Documents\\oxford\\oxfordhack2k19\\model_v6_23.hdf5")
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        face_image = cv2.resize(frame, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        predicted_class = np.argmax(model.predict(face_image))
        print(predicted_class)
        if predicted_class in [0,1,2,5,6]:
            try:

                emoji = cv2.resize(emotion_dict[predicted_class],(w-10,h-10))

                cv2.circle(frame,(x+int(w/2),y+int(h/2)),int((h -10)/2),(0,0,0),-1)

                for i in range(h - 10):
                    for j in range(w-10):
                        if frame[y+i,x+j].all() == 0:
                            frame[y+i,x+j] = emoji[i,j]


            except Exception as e:
                print(e)
                continue

    cv2.imshow('FaceDetection', frame)
    # ESC Pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()