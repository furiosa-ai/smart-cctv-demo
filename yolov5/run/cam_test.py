import argparse
import cv2

from utils.mot.video_input import VideoInput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1)
    args = parser.parse_args()

    """
    rtspsrc drop-on-latency=true latency=0 'location=rtsp://admin:!2gkakCVPR@192.168.1.101/profile2/media.smp' ! decodebin ! \
    videoscale ! video/x-raw,width=640,height=480  ! videoconvert
    """

    addr = "rtsp://admin:!2gkakCVPR@192.168.1.101/profile2/media.smp"
    # args.input[0] = f"rtspsrc location={addr} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
    # args.input[0] = f"rtspsrc location={addr} latency=0 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=true"
    args.input[0] = f"rtspsrc location={addr} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=true"
    video_input = VideoInput(args.input[0])

    while video_input.is_open():
        img = video_input()

        cv2.imshow("out", img)
        key = cv2.waitKey(1) & 0xff

        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
