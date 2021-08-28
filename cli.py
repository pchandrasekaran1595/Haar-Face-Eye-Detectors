import os
import sys
from cv2 import imshow, waitKey, destroyAllWindows, CAP_PROP_POS_FRAMES

import utils as u

#####################################################################################################

def app():
    args_1 = "--image"
    args_2 = "--video"
    args_3 = "--realtime"
    args_4 = "--mode"
    args_5 = "--name"
    args_6 = "--downscale"

    do_image, do_video, do_realtime = None, None, None
    name = None
    mode = None
    factor = None

    if args_1 in sys.argv: do_image = True
    if args_2 in sys.argv: do_video = True
    if args_3 in sys.argv: do_realtime = True
    if args_4 in sys.argv: mode = sys.argv[sys.argv.index(args_4) + 1]
    if args_5 in sys.argv: name = sys.argv[sys.argv.index(args_5) + 1]
    if args_6 in sys.argv: factor = float(sys.argv[sys.argv.index(args_6) + 1])

    assert(isinstance(mode, str))
    model = u.Model(mode=mode)

    if do_image:
        assert(isinstance(name, str))
        image = u.read(os.path.join(u.DATA_PATH, name))
    
        detections1, detections2 = model.detect(image)
        model.draw_detections(image, detections1, detections2)

        u.show(u.BGR2RGB(image), mode.capitalize() + " Detected")
    
    if do_video:
        assert(isinstance(name, str))
        cap = u.read(os.path.join(u.DATA_PATH, name), video=True)
        
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                if factor:
                    frame = u.downscale(frame, factor)

                detections1, detections2 = model.detect(frame)
                model.draw_detections(frame, detections1, detections2)

                imshow(mode.capitalize() + " Detected", frame)
                if waitKey(1) == ord("q"):
                    break
            else:
                cap.set(CAP_PROP_POS_FRAMES, 0)
        
        cap.release()
        destroyAllWindows()

    if do_realtime:
        cap = u.initCapture()
        while cap.isOpened():
            _, frame = cap.read()

            detections1, detections2 = model.detect(frame)
            model.draw_detections(frame, detections1, detections2)

            imshow(mode.capitalize() + " Detected", frame)
            if waitKey(1) == ord("q"):
                break
            
        cap.release()
        destroyAllWindows()

#####################################################################################################
