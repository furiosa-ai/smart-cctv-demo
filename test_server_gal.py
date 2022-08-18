

from utils.query_engine_base import create_gallery_cache_server


def test_face():
    create_gallery_cache_server(
        hostname="kevin-gpu",
        proj_dir_dst="~/Documents/projects/test/smart-cctv-demo",
        conda_env_name_dst="torch",
        mode="face",
        gallery_path_src="data/tom_cruise_test.mp4",
    )


def test_person():
    create_gallery_cache_server(
        hostname="kevin-gpu",
        proj_dir_dst="~/Documents/projects/test/smart-cctv-demo",
        conda_env_name_dst="torch",
        mode="person",
        gallery_path_src="data/PRW-v16.04.20/frames",
    )


def main():
    # test_face()
    test_person()


if __name__ == "__main__":
    main()
