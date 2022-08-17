import os
import sys
from types import SimpleNamespace
import cv2

from utils.mot.video_input import CapInput, VideoInput


default_args_set = SimpleNamespace(
    tab='hrnet_cm64_webcam',
    configs_yml='configs/webcam.yml',
    inputs=None,
    output_dir=None,
    interactive_vis=False,
    show_largest_person_only=False,
    show_mesh_stand_on_image=False,
    soi_camera='far',
    make_tracking=False,
    temporal_optimization=False,
    save_dict_results=False,
    save_visualization_on_img=False,
    fps_save=24,
    character='smpl',
    renderer='pyrender',
    f=None,
    model_return_loss=False,
    model_version=1,
    multi_person=True,
    new_training=False,
    perspective_proj=False,
    FOV=60,
    focal_length=443.4,
    lr=0.0003,
    adjust_lr_factor=0.1,
    weight_decay=1e-06,
    epoch=120,
    fine_tune=True,
    GPUS='-1',
    batch_size=64,
    input_size=512,
    master_batch_size=-1,
    nw=4,
    optimizer_type='Adam',
    pretrain='simplebaseline',
    fix_backbone_training_scratch=False,
    backbone='hrnet',
    model_precision='fp32',
    deconv_num=0,
    head_block_num=2,
    merge_smpl_camera_head=False,
    use_coordmaps=True,
    hrnet_pretrain='/home/furiosa/Documents/projects/workspace/warboy-ROMP/trained_models/pretrain_hrnet.pkl',
    resnet_pretrain='/home/furiosa/Documents/projects/workspace/warboy-ROMP/trained_models/pretrain_resnet.pkl',
    loss_thresh=1000,
    max_supervise_num=-1,
    supervise_cam_params=False,
    match_preds_to_gts_for_supervision=False,
    matching_mode='all',
    supervise_global_rot=False,
    HMloss_type='MSE',
    eval=False,
    eval_datasets='pw3d',
    val_batch_size=1,
    test_interval=2000,
    fast_eval_iter=-1,
    top_n_error_vis=6,
    eval_2dpose=False,
    calc_pck=False,
    PCK_thresh=150,
    calc_PVE_error=False,
    centermap_size=64,
    centermap_conf_thresh=0.25,
    collision_aware_centermap=False,
    collision_factor=0.2,
    center_def_kp=True,
    local_rank=0,
    distributed_training=False,
    distillation_learning=False,
    teacher_model_path='/export/home/suny/CenterMesh/trained_models/3dpw_88_57.8.pkl',
    print_freq=50,
    model_path='trained_models/ROMP_HRNet32_V1.pkl',
    log_path='/home/furiosa/Documents/projects/workspace/log/',
    learn_2dpose=False,
    learn_AE=False,
    learn_kp2doffset=False,
    shuffle_crop_mode=False,
    shuffle_crop_ratio_3d=0.9,
    shuffle_crop_ratio_2d=0.1,
    Synthetic_occlusion_ratio=0,
    color_jittering_ratio=0.2,
    rotate_prob=0.2,
    dataset_rootdir='/home/furiosa/Documents/projects/workspace/dataset/',
    dataset='h36m,mpii,coco,aich,up,ochuman,lsp,movi',
    voc_dir='/home/furiosa/Documents/projects/workspace/dataset/VOCdevkit/VOC2012/',
    max_person=64,
    homogenize_pose_space=False,
    use_eft=True,
    smpl_mesh_root_align=True,
    Rot_type='6D',
    rot_dim=6,
    cam_dim=3,
    beta_dim=10,
    smpl_joint_num=22,
    smpl_model_path='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/parameters',
    smpl_J_reg_h37m_path='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/parameters/J_regressor_h36m.npy',
    smpl_J_reg_extra_path='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/parameters/J_regressor_extra.npy',
    smpl_uvmap='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/parameters/smpl_vt_ft.npz',
    wardrobe='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/wardrobe',
    mesh_cloth='001',
    nvxia_model_path='/home/furiosa/Documents/projects/workspace/warboy-ROMP/model_data/characters/nvxia',
    track_memory_usage=False,
    adjust_lr_epoch=[],
    kernel_sizes=[5],
    gpu=0,
    save_mesh=False,
    save_centermap=False,
    smooth_coeff=4.0,
    visulize_platform='integrated',
    tracker='norfair',
    tracking_target='centers',
    add_trans=False,
    webcam=True,
    cam_id=0,
    multiprocess=False,
    run_on_remote_server=False,
    server_ip='localhost',
    server_port=10086
)


class ROMPRenderer:
    def __init__(self, romp_settings=None) -> None:
        romp_path = "../workspace/warboy-ROMP/"

        # del sys.path[0]
        
        del sys.modules['utils']
        del sys.modules['utils.util']
        del sys.modules['models']

        os.chdir("/home/furiosa/Documents/projects/workspace/warboy-ROMP/")
        
        sys.path.insert(0, "/home/furiosa/Documents/projects/workspace/warboy-ROMP/")

        try:
            from romp.predict.webcam_iter import Webcam_processor
        except BaseException as e:
            print(e)
        # except Exception as e:
        #     print(e)

        args_set = default_args_set

        self.romp = Webcam_processor(args_set=args_set)
        print("romp created")

    def __call__(self, image):
        outputs = self.romp(image)
        return outputs


def _test():
    romp = ROMPRenderer()

    cap = CapInput(f"/home/furiosa/Documents/projects/workspace/warboy-ROMP/r.mp4")

    while cap.is_open():
        img = cap()
        outputs = romp(img)
        cv2.imshow("romp", outputs)
        cv2.waitKey(1)


if __name__ == "__main__":
    _test()
