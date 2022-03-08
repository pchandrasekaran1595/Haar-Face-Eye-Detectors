import os
import re
import sys
import cv2
import platform
import numpy as np
import matplotlib.pyplot as plt


READ_PATH = "Files"
SAVE_PATH = "Processed"
ID, CAM_WIDTH, CAM_HEIGHT, FPS = 0, 640, 360, 30

#####################################################################################################

def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def read_file(path: str, video: bool=False) -> np.ndarray:
    if video:
        return cv2.VideoCapture(path)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)
    

def gray(image: np.ndarray, rgb: bool=False) -> np.ndarray:
    if rgb:
        return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    else:
        return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    

def BGR2RGB(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)


def downscale(image: np.ndarray, factor: float) -> np.ndarray:
    h, w, _ = image.shape
    return cv2.resize(src=image, dsize=(int(w/factor), int(h/factor)), interpolation=cv2.INTER_AREA)
    

def show(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


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

        assert re.match(r"^face$", self.mode, re.IGNORECASE) or re.match(r"^eye$", self.mode, re.IGNORECASE), "Invalid mode"

        if re.match(r"^face$", self.mode, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif re.match(r"^eye$", self.mode, re.IGNORECASE):
            self.model_1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.model_2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect(self, image):
        temp_image = gray(image.copy())
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

def app() -> None:
    args_1: tuple = ("--image", "-i")
    args_2: tuple = ("--video", "-v")
    args_3: tuple = ("--realtime", "-rt")
    args_4: tuple = ("--mode", "-m")
    args_5: tuple = ("--file", "-f")
    args_6: str = "--downscale"

    do_image: bool = False
    do_video: bool = False
    do_realtime: bool = False
    filename: str = None
    mode: str = None
    factor: float = None

    if args_1[0] in sys.argv or args_1[1] in sys.argv: do_image = True
    if args_2[0] in sys.argv or args_2[1] in sys.argv: do_video = True
    if args_3[0] in sys.argv or args_3[1] in sys.argv: do_realtime = True

    if args_4[0] in sys.argv: mode = sys.argv[sys.argv.index(args_4[0]) + 1]
    if args_4[1] in sys.argv: mode = sys.argv[sys.argv.index(args_4[1]) + 1]

    if args_5[0] in sys.argv: filename = sys.argv[sys.argv.index(args_5[0]) + 1]
    if args_5[1] in sys.argv: filename = sys.argv[sys.argv.index(args_5[1]) + 1]

    if args_6 in sys.argv: factor = float(sys.argv[sys.argv.index(args_6) + 1])

    assert(isinstance(mode, str))
    model = Model(mode=mode)

    if do_image:
        assert(isinstance(filename, str))
        image = read_file(os.path.join(READ_PATH, filename))
    
        detections1, detections2 = model.detect(image)
        model.draw_detections(image, detections1, detections2)

        show(BGR2RGB(image))
    
    if do_video:
        assert(isinstance(filename, str))
        cap = read_file(os.path.join(READ_PATH, filename), video=True)
        
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                if factor:
                    frame = downscale(frame, factor)

                detections1, detections2 = model.detect(frame)
                model.draw_detections(frame, detections1, detections2)

                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        cv2.destroyAllWindows()

    if do_realtime:
        cap = initCapture()
        while cap.isOpened():
            _, frame = cap.read()

            detections1, detections2 = model.detect(frame)
            model.draw_detections(frame, detections1, detections2)

            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) == ord("q"):
                break
            
        cap.release()
        cv2.destroyAllWindows()

#####################################################################################################


def main():
    app()


if __name__ == "__main__":
    sys.exit(main() or 0)


#####################################################################################################
