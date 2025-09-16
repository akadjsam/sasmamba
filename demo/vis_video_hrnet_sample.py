import pdb
import sys
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
# from model.MotionAGFormer_of import MotionAGFormer
# from model.MotionGraphMamba_shuffle import MotionAGFormer as MotionAGFormer_shuffle
from model.PoseMamba import PoseMamba
from model.SasMamba import SasPoseMamba
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
# from utils.data import flip_data
# from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')  # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]  # downsample is the indices of the new_indices
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample


def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


@torch.no_grad()
def get_pose3D(video_path, output_dir, modelname='motionagformer'):

    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    args2, _ = argparse.ArgumentParser().parse_known_args()
    args2.n_layers, args2.dim_in, args2.dim_feat, args2.dim_rep, args2.dim_out = 16, 3, 128, 512, 3
    args2.mlp_ratio, args2.act_layer = 4, nn.GELU
    args2.attn_drop, args2.drop, args2.drop_path = 0.0, 0.0, 0.0
    args2.use_layer_scale, args2.layer_scale_init_value, args2.use_adaptive_fusion = True, 0.00001, True
    args2.num_heads, args2.qkv_bias, args2.qkv_scale = 8, False, None
    args2.hierarchical = False
    args2.use_temporal_similarity, args2.neighbour_num, args2.temporal_connection_len = True, 2, 1
    args2.use_tcn, args2.graph_only = False, False
    args2.n_frames = 243
    args2.shuffle_rate = 0.4
    args2 = vars(args2)


    if modelname == 'motionagformer':
        ## Reload
        model = nn.DataParallel(MotionAGFormer(**args)).cuda()

        # Put the pretrained model of MotionAGFormer in 'checkpoint/'
        model_path = sorted(glob.glob(os.path.join('weights', 'motionagformer-b-h36m.pth.tr')))[0]

        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'motionagformer_shuffle':
        ## Reload
        model = nn.DataParallel(MotionAGFormer_shuffle(**args2)).cuda()

        # Put the pretrained model of MotionAGFormer in 'checkpoint/'
        # model_path = sorted(glob.glob(os.path.join('weights', 'motionagformer-b-h36m.pth.tr')))[0]
        # best base model:
        model_path = sorted(glob.glob(os.path.join('checkpoint', '20241011_181511best_epoch.pth.tr')))[0]

        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'SasPoseMamba':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.num_frame, args.embed_dim_ratio, args.mlp_ratio, args.depth, args.in_chans, args.ssm_conv = 243, 64, 2.0, 5, 2, 3
        # num_frame = 9, num_joints = 17, in_chans = 2, ssm_conv = 3, embed_dim_ratio = 256, depth = 6
        args = vars(args)
        model = nn.DataParallel(SasPoseMamba(**args)).cuda()
        # model_path = sorted(glob.glob(os.path.join('weights', 'SasPoseMamba.pth')))[0]
        model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn/best_epoch.pth.tr'
        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'SasPoseMamba_slatest':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.num_frame, args.embed_dim_ratio, args.mlp_ratio, args.depth, args.in_chans, args.ssm_conv = 243, 64, 2.0, 5, 2, 3
        # num_frame = 9, num_joints = 17, in_chans = 2, ssm_conv = 3, embed_dim_ratio = 256, depth = 6
        args = vars(args)
        model = nn.DataParallel(SasPoseMamba(**args)).cuda()
        # model_path = sorted(glob.glob(os.path.join('weights', 'SasPoseMamba.pth')))[0]
        model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn/latest_epoch.pth.tr'
        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'SasPoseMamba_L':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.num_frame, args.embed_dim_ratio, args.mlp_ratio, args.depth, args.in_chans, args.ssm_conv = 243, 128, 2.0, 10, 2, 3
        # num_frame = 9, num_joints = 17, in_chans = 2, ssm_conv = 3, embed_dim_ratio = 256, depth = 6
        args = vars(args)
        model = nn.DataParallel(SasPoseMamba(**args)).cuda()
        # model_path = sorted(glob.glob(os.path.join('weights', 'SasPoseMamba.pth')))[0]
        model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn_large/best_epoch.pth.tr'
        # model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn_large/latest_epoch.pth.tr'
        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'SasPoseMamba_Latest':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.num_frame, args.embed_dim_ratio, args.mlp_ratio, args.depth, args.in_chans, args.ssm_conv = 243, 128, 2.0, 10, 2, 3
        # num_frame = 9, num_joints = 17, in_chans = 2, ssm_conv = 3, embed_dim_ratio = 256, depth = 6
        args = vars(args)
        model = nn.DataParallel(SasPoseMamba(**args)).cuda()
        # model_path = sorted(glob.glob(os.path.join('weights', 'SasPoseMamba.pth')))[0]
        # model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn_large/best_epoch.pth.tr'
        model_path = 'checkpoint/36m/PoseMamba_243_ST_dcn_large/latest_epoch.pth.tr'
        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    # ## Reload
    # model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    #
    # # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    # model_path = sorted(glob.glob(os.path.join('weights', 'motionagformer-b-h36m.pth.tr')))[0]
    #
    # pre_dict = torch.load(model_path)
    # model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)

    clips, downsample = turn_into_clips(keypoints)  # clips: [N, F, 17, 3], downsample: [243]

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        input_2D = keypoints[0][i]

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = output_dir + 'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)

    print('\nGenerating 3D pose...')
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        # print(input_2D.shape)
        # pdb.set_trace()
        if modelname == 'SasPoseMamba' or modelname == 'PoseMamba' or modelname == 'PoseMamba_latest' or modelname == 'SasPoseMamba_L' or modelname == 'SasPoseMamba_Latest' or modelname == 'SasPoseMamba_slatest':
            input_2D = input_2D[:, :, :, :2]  # (N, T, 17, 2)
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        for j, post_out in enumerate(post_out_all):
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir + 'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d' % (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d' % (idx * 243 + j))) + '_3D.png', dpi=200, format='png',
                        bbox_inches='tight')
            plt.close(fig)

    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\n Crop 2D Figs...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(7.5, 5.4))
        ax = plt.subplot(111)
        showimage(ax, image_2d)
        ax.set_title(f"Frames{i}", fontsize=font_size)
        ## save
        output_dir_pose = output_dir + 'pose2Dcrop/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose2Dcrop.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    print('\n Crop 3D Figs...')
    for i in tqdm(range(len(image_3d_dir))):
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(7.5, 5.4))
        ax = plt.subplot(111)
        showimage(ax, image_3d)
        ax.set_title(f"Frames{i}", fontsize=font_size)
        ## save
        output_dir_pose = output_dir + 'pose3Dcrop/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose3Dcrop.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    print('Crop 3D pose successful!')
    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        ## save
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

@torch.no_grad()
def viz_36M(output_dir, modelname='motionagformer'):
    # model preparation
    # motionagformer-b-h36m.pth.tr
    if modelname == 'motionagformer':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
        args.mlp_ratio, args.act_layer = 4, nn.GELU
        args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
        args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
        args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
        args.hierarchical = False
        args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
        args.use_tcn, args.graph_only = False, False
        args.n_frames = 243
        args = vars(args)
        ## Reload
        model = nn.DataParallel(MotionAGFormer(**args)).cuda()
        # Put the pretrained model of MotionAGFormer in 'checkpoint/'
        model_path = sorted(glob.glob(os.path.join('weights', 'motionagformer-b-h36m.pth.tr')))[0]

        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    elif modelname == 'motionagformer_shuffle':
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
        args.mlp_ratio, args.act_layer = 4, nn.GELU
        args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
        args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
        args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
        args.hierarchical = False
        args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
        args.use_tcn, args.graph_only = False, False
        args.n_frames = 243
        args.shuffle_rate = 0.4
        args = vars(args)
        ## Reload
        model = nn.DataParallel(MotionAGFormer_shuffle(**args)).cuda()
        # Put the pretrained model of MotionAGFormer in 'checkpoint/'
        model_path = sorted(glob.glob(os.path.join('weights', 'motionagformer-b-h36m.pth.tr')))[0]
        pre_dict = torch.load(model_path)
        model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()


    # 3.6M dataset 2D pose
    # parames :
    # Data  preparation:
    print("Data preparation...")
    dataargs, _ = argparse.ArgumentParser().parse_known_args()
    dataargs.data_root = 'data/motion3d/'
    dataargs.data_root_2d = 'data/motion2d/'
    dataargs.subset_list = ['H36M-243']
    dataargs.dt_file = 'h36m_sh_conf_cam_source_final.pkl'
    dataargs.num_joints = 17
    dataargs.root_rel = True  # Normalizing joints relative to the root joint
    dataargs.add_velocity = False
    dataargs.flip = True
    dataargs.use_proj_as_2d = False
    common_loader_params = {
        'batch_size': 16,
        # 'num_workers': 0,
        'num_workers': 6 - 1,
        'pin_memory': True,
        # 'prefetch_factor': (opts.num_cpus - 1) // 3,
        'prefetch_factor': 3,
        'persistent_workers': False
    }
    # train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(dataargs, dataargs.subset_list, 'test')
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)
    sample = next(iter(test_loader))
    print(len(sample))
    print("Data preparation successful!")
    pdb.set_trace()

    ## input
    videpred = True
    if videpred:
        keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
        # keypoints = np.load('demo/lakeside3.npy')
        # keypoints = keypoints[:240]
        # keypoints = keypoints[None, ...]
        # keypoints = turn_into_h36m(keypoints)

        clips, downsample = turn_into_clips(keypoints)  # clips: [N, F, 17, 3], downsample: [243]

        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ## 3D
        print('\nGenerating 2D pose image...')
        for i in tqdm(range(video_length)):
            ret, img = cap.read()
            if img is None:
                continue
            img_size = img.shape

            input_2D = keypoints[0][i]

            image = show2Dpose(input_2D, copy.deepcopy(img))

            output_dir_2D = output_dir + 'pose2D/'
            os.makedirs(output_dir_2D, exist_ok=True)
            cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)
    else:
        # 3.6m dataset 2D pose
        keypoints = np.load(output_dir + 'keypoints.npz', allow_pickle=True)['reconstruction']




    print('\nGenerating 3D pose...')
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        for j, post_out in enumerate(post_out_all):
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir + 'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d' % (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d' % (idx * 243 + j))) + '_3D.png', dpi=200, format='png',
                        bbox_inches='tight')
            plt.close(fig)

    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        ## save
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    print("From video to 2D and 3D pose...")
    parser = argparse.ArgumentParser()
    # sample_video1.mp4, sample_video2.mp4, sample_video3.mp4
    parser.add_argument('--video', type=str, default='sample_video5.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    # modelname = 'motionagformer'
    # modelname = 'motionagformer_shuffle'
    # modelname = 'motionagformer'
    modelname = 'SasPoseMamba'
    # modelname = 'SasPoseMamba_L'
    # modelname = 'SasPoseMamba_Latest'
    # modelname = 'SasPoseMamba_slatest'

    output_dir = './demo/output/' + modelname +'/'+ video_name + '/'
    get_pose2D(video_path, output_dir)

    get_pose3D(video_path, output_dir, modelname)
    img2video(video_path, output_dir)
    print('Generating demo successful!')

# if __name__ == "__main__":
#     print("Visualizing 3.6M dataset...")
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--video', type=str, default='sample_video3.mp4', help='input video')
#     parser.add_argument('--gpu', type=str, default='0', help='input video')
#     args = parser.parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     modelname = 'motionagformer'
#     dataset = 'H36M_243'
#     # modelname = 'motionagformer_shuffle'
#     output_dir = './demo/output/' + modelname +'/'+ dataset + '/'
#     viz_36M(output_dir, modelname)
#     # get_pose2D_36m(output_dir)
#     # get_pose3D_36m(output_dir, modelname)
#     # img2video(video_path, output_dir)
#     print('Generating demo successful!')

