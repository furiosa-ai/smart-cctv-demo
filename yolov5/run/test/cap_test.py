from utils.mot.video_input import CapInput
import cv2


cap = CapInput("datasets/MOT20/vid/MOT20-02-640.mp4", loop_video=True)
while True:
    cv_img = cap()
    cv2.imshow("out", cv_img)
    cv2.waitKey(1)
