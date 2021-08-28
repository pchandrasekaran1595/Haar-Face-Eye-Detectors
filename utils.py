import os
import re
import cv2
import platform
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

import utils as u

#####################################################################################################

os.system("color")
def myprint(text: str, color="white"):
    print(colored(text=text, color=color))


def breaker(num=50, char="*"):
    myprint("\n" + num*char + "\n", "magenta")

#####################################################################################################

def read(path: str, video=False) -> np.ndarray:
    if video:
        return cv2.VideoCapture(path)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)


def show(image: np.ndarray, title: None) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def gray(image: np.ndarray, rgb=False) -> np.ndarray:
    if rgb:
        return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    else:
        return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)


def BGR2RGB(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)


def downscale(image: np.ndarray, factor: int) -> np.ndarray:
    h, w, _ = image.shape
    return cv2.resize(src=image, dsize=(int(w/factor), int(h/factor)), interpolation=cv2.INTER_AREA)


def initCapture():
    if platform.system() != "Windows":
        cap = cv2.VideoCapture(ID)
    else:
        cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    return cap

#####################################################################################################

class Model(object):
    def __init__(self, mode="face"):
        self.mode = mode

        if re.match(r"face", self.mode, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif re.match(r"eye", self.mode, re.IGNORECASE):
            self.model_1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.model_2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect(self, image):
        temp_image = u.gray(image.copy())
        if re.match(r"face", self.mode, re.IGNORECASE):
            detections = self.model.detectMultiScale(image=temp_image)
            return detections, None
        elif re.match(r"eye", self.mode, re.IGNORECASE):
            eye_detections = None
            face_detections = self.model_1.detectMultiScale(image=temp_image)
            for (x, y, w, h) in face_detections:
                roi_image = gray(image[y:y+h, x:x+w].copy())
                eye_detections = self.model_2.detectMultiScale(image=roi_image)
            return face_detections, eye_detections
    
    def draw_detections(self, image, detections1, detections2=None):
        if detections2 is None:
            for (x, y, w, h) in detections1:
                cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
        else:
            for (x1, y1, w1, h1) in detections1:
                for (x2, y2, w2, h2) in detections2:
                    cv2.rectangle(img=image[y1:y1+h1, x1:x1+w1], pt1=(x2, y2), pt2=(x2+w2, y2+h2), color=(255, 0, 0), thickness=2)
                cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x1+w1, y1+h1), color=(0, 255, 0), thickness=2)

        
#####################################################################################################

DATA_PATH = "./Files"
CAM_WIDTH, CAM_HEIGHT, FPS, ID = 640, 360, 30, 0

#####################################################################################################
