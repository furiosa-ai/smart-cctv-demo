import numpy as np
import cv2
import skimage


# TODO: flip image and normalize emb
class FaceRecognitionPipeline:
    def __init__(self, face_det, face_rec_pred, database) -> None:
        self.face_det = face_det
        self.face_rec_pred = face_rec_pred
        self.database = database

    def warp_face(self, img, landmark5):
        SRC = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]]
            , dtype=np.float32)
        SRC[:, 0] += 8.0

        st = skimage.transform.SimilarityTransform()
        st.estimate(landmark5, SRC)
        img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        return img

    def warp(self, img, landmarks):
        # first face only
        facial5points = np.array(
                    [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)],
                    dtype=np.float32)

        return self.warp_face(img, facial5points)

    def __call__(self, img, warp_img=True):
        bounding_boxes, conf, landmarks = zip(*self.face_det(img))
        bounding_boxes = bounding_boxes[:1]
        landmarks = landmarks[:1]

        if warp_img:
            img = self.warp(img, landmarks)
        feat = self.face_rec_pred(img)
        feat = feat.reshape(-1)
        
        sim_faces = []

        if self.database is not None and len(self.database) > 0:
            closest = self.database.find_closest(feat, 5)
            for idx, sim in zip(*closest):
                sim_faces.append({
                    "name": self.database.get_name(idx),
                    "img": self.database.get_image(idx),
                    "sim": sim,
                    # "sim_deg": np.rad2deg(sim),
                    "box": bounding_boxes[0],
                    "lm": landmarks[0],
                })

        return {
            "box": bounding_boxes[0] if len(bounding_boxes) > 0 else None,
            "lm": landmarks[0] if len(landmarks) > 0 else None,
            "feat": feat,
            "sim_faces": sim_faces,
        }