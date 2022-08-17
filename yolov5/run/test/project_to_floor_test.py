import numpy as np
import cv2
from calibration.util.camera_calib_node import CameraCalibrationNode


def project_to_floor(frame, img_p, out_size): 
    # w = int(np.linalg.norm(np.array(right_top) - np.array(left_top)))
    # h = int(np.linalg.norm(np.array(left_bottom) - np.array(left_top)))
    w, h = out_size
    
    # Coordinates that you want to Perspective Transform
    pts1 = np.float32(img_p)
    # Size of the Transformed Image
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    
    print(f"pst1: {pts1}")
    print(f"pst2: {pts2}")
    
    # for val in pts1:
    #     cv2.circle(frame,(val[0],val[1]),5,(0,0,255),-1)

    M = cv2.getPerspectiveTransform(pts1,pts2)

    frame = np.concatenate([frame, np.full((frame.shape[0], frame.shape[1], 1), 1, dtype=np.uint8)], 2)
    dst = cv2.warpPerspective(frame,M,(int(w),int(h)))
    
    # cv2.imshow("paper", frame)
    # cv2.imshow("dst", dst)
    
    # cv2.waitKey(0)

    return dst

# p = np.random.rand(3,4)
# H = get_inverse_pespective(p)
# src_point = (5,10)
# dst_point = project_to_floor(src_point, H)


def proj_calib(calib_name):
    img_file = f"cam_calib_data/fur15/cam_calib{calib_name}/1920_1080/extr_img.png"
    calib_file = f"cfg/calib/fur15/cam_calib{calib_name}.yaml"

    img = cv2.imread(img_file)
    calib_node = CameraCalibrationNode.from_file(calib_file)
    calib_node.undistort_camera()
    # img = calib_node.process_image(img)  # ext intr image is already undistorted

    x1, y1, x2, y2 = -10, -10, 10, 10

    obj_points = np.array([
        [x1, y1, 0],
        [x2, y1, 0],
        [x1, y2, 0],
        [x2, y2, 0],
    ], dtype=float)

    w = 2048
    h = round((y2 - y1) / (x2 - x1) * w)

    img_p = calib_node.project_points(obj_points)

    return project_to_floor(img, img_p, (w, h)), img


def fuse_img(imgs):
    grid = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], imgs[3]])])

    imgs = [img.astype(float) for img in imgs]
    masks = [img[:, :, 3] != 0 for img in imgs]

    blend = np.sum(imgs, 0)

    mask = blend[:, :, 3] != 0
    num_overlap = np.sum(masks, 0)

    blend = blend[:, :, :3]
    blend /= np.maximum(num_overlap, 1)[:, :, None]
    # blend[mask] /= 4

    blend = blend.astype(np.uint8)

    return grid, blend


def main():
    calibs = [101, 102, 103, 105]

    imgs, imgs_unproj = zip(*[proj_calib(c) for c in calibs])

    grid, blend = fuse_img(imgs)
    grid_unproj = np.vstack([np.hstack([imgs_unproj[0], imgs_unproj[1]]), np.hstack([imgs_unproj[2], imgs_unproj[3]])])
    
    grid = cv2.resize(grid, None, fx=0.25, fy=0.25)
    grid_unproj = cv2.resize(grid_unproj, None, fx=0.25, fy=0.25)
    cv2.imshow("grid", grid)
    cv2.imshow("blend", blend)
    cv2.imshow("grid unproj", grid_unproj)
    # cv2.imshow("blend unproj", blend_unproj)
    cv2.waitKey(0)


    cv2.imwrite("cam_calib_data/fur15/proj.jpg", blend)


if __name__ == "__main__":
    main()
