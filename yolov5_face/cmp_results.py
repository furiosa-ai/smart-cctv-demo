import cv2
import pickle


def show_results(img, xyxy, conf, landmarks, color):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    cv2.rectangle(img, (x1,y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

    # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, color, -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def main():
    img_file = "data/images/test.jpg"

    files = [
        "results/onnx/result.pkl",
        # "results/trt/result.pkl",
        # "results/cpu_relu/result.pkl",
        # "results/warboy_relu/result.pkl",
        "results/warboy/result.pkl"
    ]

    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

    outputs = [pickle.load(open(f, "rb")) for f in files]

    colors = [
        (0, 255, 0),
        (255, 0, 0),
    ]

    for i, out in enumerate(outputs):
        for xyxy, conf, landmarks in out:
            img = show_results(img, xyxy, conf, landmarks, colors[i])

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('cmp_cpu_warboy.jpg', img)


if __name__ == '__main__':
    main()
