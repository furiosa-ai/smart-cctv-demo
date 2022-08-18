from pathlib import Path
import os
import subprocess
from ai_util.dataset import ImageDataset


class GalleryCacheBuilderRemote:
    def __init__(self, hostname, proj_dir_dst, conda_env_name_dst) -> None:
        self.hostname = hostname
        self.proj_dir_dst = proj_dir_dst
        self.conda_env_name_dst = conda_env_name_dst

    def __call__(self, mode, gallery_path_src):
        return create_gallery_cache_server(self.hostname, self.proj_dir_dst, self.conda_env_name_dst, mode, gallery_path_src)


def create_gallery_cache_server(hostname, proj_dir_dst, conda_env_name_dst, mode, gallery_path_src):
    def _ssh_cmd(hostname, cmds):
        return ["ssh", hostname, " && ".join(
            (" ".join([str(ele) for ele in cmd])) for cmd in cmds
        )]

    if mode == "face":
        sub_proj_dir = "insightface"
        script_path = "recognition/arcface_torch/query_video.py"
    elif mode == "person":
        sub_proj_dir = "torchreid"
        script_path = "query_video.py"
    else:
        raise Exception(mode)

    gallery_path_src = Path(gallery_path_src)
    proj_dir_dst = Path(proj_dir_dst)
    gallery_path_dst = proj_dir_dst / "tmp" / (mode + "_" + str(gallery_path_src).replace("/", "_"))

    gallery_cache_name = os.path.abspath(gallery_path_src).replace("/", "_")

    gallery_cache_src = Path() / "galleries" / mode / (gallery_cache_name + ".npz")
    gallery_cache_dst = proj_dir_dst / gallery_cache_src

    assert gallery_path_src.exists()
    cp_suffix = "/" if gallery_path_src.is_dir() else ""

    if not gallery_cache_src.is_file():
        gallery_cache_src.parent.mkdir(parents=True, exist_ok=True)
        cmds = [
            ["rsync", "-avz", str(gallery_path_src) + cp_suffix, hostname + ":" + str(gallery_path_dst)],  # copy data to server
            _ssh_cmd(hostname, [
                ["source", "~/anaconda3/etc/profile.d/conda.sh"],
                ["conda", "activate", conda_env_name_dst],
                ["cd", proj_dir_dst / sub_proj_dir],
                ["(", "killall", "python", "||", "true", ")"],
                ["python", script_path, "--gallery", gallery_path_dst, "--gallery_name", gallery_cache_name, "--device", "furiosa"],
            ]),
            ["rsync", "-avz", hostname + ":" + str(gallery_cache_dst), str(gallery_cache_src)],  # copy gallery cache to client
        ]

        # print(" && \\\n".join(
        #     " ".join(cmd) for cmd in cmds
        # ))
        # return

        print("Sending data to server")
        res = subprocess.call(cmds[0]) 
        if res != 0:
            print("Failed")
            return None

        print("Computing gallery cache")
        res = subprocess.call(cmds[1])
        if res != 0:
            print("Failed")
            return None

        print("Receiving computed gallery cache")
        res = subprocess.call(cmds[2])
        if res != 0:
            print("Failed")
            return None

    return gallery_cache_name


class QueryEngineBase:
    def __init__(self, topk=5) -> None:
        self.topk = topk
        self.vis_best_only = False

    def set_gallery_data(self, path):
        self.gallery_data = ImageDataset(path, limit=None, frame_step=1)
        self.process_gallery_data()
        return self.gallery_data

    def process_gallery_data(self):
        raise NotImplementedError

    def create_query_from_image(self, img):
        raise NotImplementedError

    def query(self, query, topk):
        raise NotImplementedError

