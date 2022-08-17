from turtle import update
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import time
import threading


class _Viewer:
    def __init__(self) -> None:
        self.v = None

        fuze_trimesh = trimesh.load('../data/mesh/pyrender_examples/models/fuze.obj')
        # Obj_trimesh = trimesh.primitives.Cylinder(radius=0.004, height=0.21)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        scene = pyrender.Scene()
        mesh_node = scene.add(mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2)/2
        camera_pose = np.array([
        [0.0, -s,   s,   0.3],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  s,   s,   0.35],
        [0.0,  0.0, 0.0, 1.0],
        ])
        """
        camera_pose = np.array([
            [1.0,  0.0,   0.0,   0.0],
            [0.0,  1.0, 0.0, 0.0],
            [0.0,  0.0,   1.0,   0.35],
            [0.0,  0.0, 0.0, 1.0],
        ])

        """
        scene.add(camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
        scene.add(light, pose=camera_pose)
        # r = pyrender.OffscreenRenderer(400, 400)

        v = pyrender.Viewer(scene, auto_start=False)

        self.scene = scene
        self.mesh_node = mesh_node
        self.v = v

        t = threading.Thread(target=self._update_thread)
        t.start()

        v.start()
        t.join()

    def _update_thread(self):
        # while self.v is None:
        #     time.sleep(0.1)

        i = 0
        while True:
            pose = np.eye(4)
            pose[:3,3] = [0, i, 0]
            self.v.render_lock.acquire()
            self.scene.set_pose(self.mesh_node, pose)
            self.v.render_lock.release()
            i += 0.001
            time.sleep(0.1)


def main():
    _Viewer()


if __name__ == "__main__":
    main()
