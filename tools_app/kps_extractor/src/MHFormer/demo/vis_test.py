import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
import time # 引入时间模块进行测速

sys.path.append(os.getcwd())
from model.mhformer import Model
from common.camera import *

def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('\n[Step 1/2] Generating 2D pose (HRNet)...')
    start_time = time.time() # 开始计时 2D 阶段
    
    with torch.no_grad():
        # 提取 2D 关键点
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=False) # gen_output 改为 False 减少不必要的输出
    
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    
    end_time = time.time() # 结束计时 2D 阶段
    cost_time = end_time - start_time

    output_dir_2d = output_dir + 'input_2D/'
    os.makedirs(output_dir_2d, exist_ok=True)
    output_npz = output_dir_2d + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    
    return video_length, cost_time


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/351'
    args.n_joints, args.out_joints = 17, 17

    ## 加载模型
    model = Model(args).cuda()
    model_dict = model.state_dict()
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    model.eval()

    ## 读取 2D 输入
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    # 提前获取视频宽高，避免在 3D 推理循环中解码视频！(极速优化)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print('\n[Step 2/2] Generating 3D pose (MHFormer inference)...')
    output_3d_all = []
    
    start_time = time.time() # 开始计时 3D 阶段
    
    with torch.no_grad(): # 确保不计算梯度，加速推理
        for i in tqdm(range(video_length)):
            ## 截取时间窗口
            start = max(0, i - args.pad)
            end =  min(i + args.pad, len(keypoints[0])-1)
            input_2D_no = keypoints[0][start:end+1]
            
            left_pad, right_pad = 0, 0
            if input_2D_no.shape[0] != args.frames:
                if i < args.pad:
                    left_pad = args.pad - i
                if i > len(keypoints[0]) - args.pad - 1:
                    right_pad = i + args.pad - (len(keypoints[0]) - 1)
                input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
            
            joints_left =  [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]

            # 直接使用提前读取的宽高
            input_2D = normalize_screen_coordinates(input_2D_no, w=img_w, h=img_h)  

            input_2D_aug = copy.deepcopy(input_2D)
            input_2D_aug[ :, :, 0] *= -1
            input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
            input_2D = input_2D[np.newaxis, :, :, :, :]
            input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

            ## 网络推理
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip     = model(input_2D[:, 1])

            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            output_3D = output_3D[0:, args.pad].unsqueeze(1) 
            output_3D[:, :, 0, :] = 0
            post_out = output_3D[0, 0].cpu().detach().numpy()

            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            
            output_3d_all.append(post_out)
            # 删除了所有的绘图和截图代码！
            
    end_time = time.time() # 结束计时 3D 阶段
    cost_time = end_time - start_time
            
    ## 保存 3D 关键点矩阵
    output_3d_all = np.stack(output_3d_all, axis = 0)
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    return cost_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='gpu index')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    print(f"========== 性能基准测试开始 (纯推理) ==========")
    print(f"Target Video: {args.video}")
    
    total_start = time.time()
    
    # 运行并测速
    frames, time_2d = get_pose2D(video_path, output_dir)
    time_3d = get_pose3D(video_path, output_dir)
    
    total_time = time.time() - total_start

    # 打印硬核报告
    print("\n========== 🚀 测速报告 (RTX 4050 纯算力) ==========")
    print(f"视频总帧数: {frames} frames")
    print(f"--------------------------------------------------")
    print(f"[阶段 1] 2D 前端侦测 (HRNet):")
    print(f"  - 耗时: {time_2d:.2f} 秒")
    print(f"  - 速度: {frames / time_2d:.2f} FPS")
    print(f"--------------------------------------------------")
    print(f"[阶段 2] 3D 后端推理 (MHFormer):")
    print(f"  - 耗时: {time_3d:.2f} 秒")
    print(f"  - 速度: {frames / time_3d:.2f} FPS")
    print(f"--------------------------------------------------")
    print(f"⚡ 总耗时 (含IO): {total_time:.2f} 秒")
    print(f"⚡ 综合流水线 FPS: {frames / total_time:.2f} FPS")
    print(f"==================================================")
    print("生成完毕！纯坐标数据已保存在 demo/output/ 目录下。")