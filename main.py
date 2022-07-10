import os
import re
import sys
import cv2
import platform
import numpy as np
import matplotlib.pyplot as plt


BASE_PATH: str   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH: str  = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH: str = os.path.join(BASE_PATH, 'output')

ID: int = 0
CAM_WIDTH: int  = 640
CAM_HEIGHT: int = 360 
FPS: int = 30


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def read_file(path: str, video: bool=False) -> np.ndarray:
    if video: return cv2.VideoCapture(path)
    else: return cv2.imread(path, cv2.IMREAD_COLOR)
    

def gray(image: np.ndarray, rgb: bool=False) -> np.ndarray:
    if rgb: return cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    else: return cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    

def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def draw_detections(image: np.ndarray, face_detections: tuple, eye_detections: tuple=None):
    if eye_detections is None:
        for (x, y, w, h) in face_detections:
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
    else:
        for (x1, y1, w1, h1) in face_detections:
            for (x2, y2, w2, h2) in eye_detections:
                cv2.rectangle(img=image[y1:y1+h1, x1:x1+w1], pt1=(x2, y2), pt2=(x2+w2, y2+h2), color=(255, 0, 0), thickness=2)
            cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x1+w1, y1+h1), color=(0, 255, 0), thickness=2)


class Model(object):
    def __init__(self, model_type="face"):
        self.model_type = model_type

        assert re.match(r"^face$", self.model_type, re.IGNORECASE) \
            or re.match(r"^eye$", self.model_type, re.IGNORECASE), "Invalid mode"

        if re.match(r"^face$", self.model_type, re.IGNORECASE):
            self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        elif re.match(r"^eye$", self.model_type, re.IGNORECASE):
            self.model_1 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.model_2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect(self, image):
        temp_image = gray(image.copy())
        if re.match(r"face", self.model_type, re.IGNORECASE):
            detections = self.model.detectMultiScale(image=temp_image)
            return detections, None
        elif re.match(r"eye", self.model_type, re.IGNORECASE):
            eye_detections = None
            face_detections = self.model_1.detectMultiScale(image=temp_image)
            for (x, y, w, h) in face_detections:
                roi_image = gray(image[y:y+h, x:x+w].copy())
                eye_detections = self.model_2.detectMultiScale(image=roi_image)
            return face_detections, eye_detections


def main():
    args_1: tuple = ("--mode", "-m")
    args_2: tuple = ("--model", "-mo")
    args_3: tuple = ("--filename", "-f")
    args_4: tuple = ("--downscale", "-ds")
    args_5: tuple = ("--save", "-s")

    mode: str = "image"
    model_type: str = "face"
    filename: str = "Test_1.jpg"
    downscale: float = None
    save: bool = False

    if args_1[0] in sys.argv: mode = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: mode = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: model_type = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: filename = sys.argv[sys.argv.index(args_3[0]) + 1]
    if args_3[1] in sys.argv: filename = sys.argv[sys.argv.index(args_3[1]) + 1]

    if args_4[0] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_4[0]) + 1])
    if args_4[1] in sys.argv: downscale = float(sys.argv[sys.argv.index(args_4[1]) + 1])

    if args_5[0] in sys.argv or args_5[1] in sys.argv: save = True


    model = Model(model_type=model_type)


    if re.match(r"image", mode, re.IGNORECASE):

        assert filename in os.listdir(INPUT_PATH), "File not Found"

        image = read_file(os.path.join(INPUT_PATH, filename))
    
        detections1, detections2 = model.detect(image)

        if save: 
            pass
        else: 
            disp_image = image.copy()
            draw_detections(disp_image, detections1, detections2)
            show_image(image=disp_image, title="Detections")
    
    elif re.match(r"video", mode, re.IGNORECASE):

        assert filename in os.listdir(INPUT_PATH), "File not Found"

        cap = read_file(os.path.join(INPUT_PATH, filename), video=True)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if downscale:
                    frame = cv2.resize(src=frame, dsize=(int(frame.shape[1]/downscale), int(frame.shape[0]/downscale)), interpolation=cv2.INTER_AREA)

                detections1, detections2 = model.detect(frame)
                draw_detections(frame, detections1, detections2)

                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        cv2.destroyAllWindows()

    elif re.match(r"realtime", mode, re.IGNORECASE):

        if platform.system() != "Windows":
            cap = cv2.VideoCapture(ID)
        else:
            cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)
        
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        while cap.isOpened():
            _, frame = cap.read()

            detections1, detections2 = model.detect(frame)
            draw_detections(frame, detections1, detections2)

            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): 
                break
            
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("\n --- Unknown Mode ---\n".upper())


if __name__ == "__main__":
    sys.exit(main() or 0)


