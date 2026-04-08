# run_imputation.py
import argparse
from multiprocessing.spawn import prepare
import os, sys
import numpy as np
import torch

import pickle
import os.path as osp

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # OmniControl/
sys.path.append(str(ROOT))
os.chdir(str(ROOT))
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_compare_3d_motion
import shutil
from data_loaders.tensors import collate
from utils.text_control_example import collate_all
from os.path import join as pjoin

import time

# 引入你的适配器 (假设你保存为 utils/skeleton_adapter.py)
# 如果没有分开保存，可以直接把 Adapter 类粘贴在脚本最上方
class SkeletonAdapter:
    def __init__(self):
        self.h36m_structure_dict = {
               0:"Bottom_torso", 1 : "L_Hip", 2 : "L_Knee", 3 : "L_Foot",
                4:"R_Hip", 5 : "R_Knee", 6 : "R_Foot", 7 : "Center_torso",
                8:"Upper_torso", 9 : "Neck", 10 : "Head", 11 : "R_Shoulder",
                12:"R_Elbow", 13 : "R_Hand", 14 : "L_Shoulder", 15 : "L_Elbow",
                16:"L_Hand",
        }
        
        self.smpl_structure_dict = {
               0:"Pelvis", 1 : "L_Hip", 2 : "R_Hip", 3 : "Spine1",
                4:"L_Knee", 5 : "R_Knee", 6 : "Spine2", 7 : "L_Ankle",
                8:"R_Ankle", 9 : "Spine3", 10 : "L_Toe", 11 : "R_Toe",
                12:"Neck", 13: "L_Collar", 14 : "R_Collar", 15 : "Head",
                16:"L_Shoulder", 17 : "R_Shoulder", 18 : "L_Elbow",
                19:"R_Elbow", 20 : "L_Hand", 21 : "R_Hand",
        }
        pass

    def h36m_to_smpl22(self, h36m_data):
        """
        将 Human3.6M (17点, Y-Down) -> SMPL (22点, Y-Up)
        
        参数:
            h36m_data: (Frames, 17, 3)
        返回:
            smpl_data: (Frames, 22, 3)
        """
        # 1. 坐标系转换 (Fix Coordinate System)
        # 复制一份以免修改原数据
        h36m_data = h36m_data.copy()
        
        # 归一化/去中心化: 将骨盆 (Index 0) 移到原点
        # 这一步非常重要，因为两个数据的绝对Z值(深度)差异很大
        root = h36m_data[:, 0:1, :].copy()
        h36m_data = h36m_data - root
        
        #* 翻转 Y 轴: H36M是Y向下，SMPL是Y向上
        h36m_data[:, :, 1] *= -1
        
        # 2. 关节点映射 (Joint Mapping)
        frames = h36m_data.shape[0]
        smpl_data = np.zeros((frames, 22, 3), dtype=h36m_data.dtype)
        
        # --- 直接映射 ---
        # Body
        smpl_data[:, 0] = h36m_data[:, 0]   # Pelvis
        smpl_data[:, 3] = h36m_data[:, 7]   # Spine1 <- Center torso
        
        if h36m_data[:, 7].all() == 0.0 or h36m_data[:, 8].all() == 0.0:
            smpl_data[:, 6] = 0.0
        else:
            smpl_data[:, 6] = (h36m_data[:, 7] + h36m_data[:, 8]) / 2.0 # Spine2 (插值)
        smpl_data[:, 9] = h36m_data[:, 8]   # Spine3 <- Upper torso
        smpl_data[:, 12] = h36m_data[:, 9]  # Neck
        smpl_data[:, 15] = h36m_data[:, 10] # Head
        
        # Legs (H36M: Hip->Knee->Foot) -> SMPL (Hip->Knee->Ankle)
        smpl_data[:, 1] = h36m_data[:, 1]   # L_Hip
        smpl_data[:, 4] = h36m_data[:, 2]   # L_Knee
        smpl_data[:, 7] = h36m_data[:, 3]   # L_Ankle
        
        smpl_data[:, 2] = h36m_data[:, 4]   # R_Hip
        smpl_data[:, 5] = h36m_data[:, 5]   # R_Knee
        smpl_data[:, 8] = h36m_data[:, 6]   # R_Ankle
        
        # L_Foot/Toe (Index 10) & R_Foot/Toe (Index 11):
        # H36M 没有脚尖数据。我们暂时将其设置为与 Ankle 重合，或者稍微向下延伸一点。
        # 为了 Inpainting 方便，建议设为与 Ankle 重合，但在 Mask 中将其标记为"缺失"，让模型去生成。
        # smpl_data[:, 10] = smpl_data[:, 7] # L_Toe = L_Ankle
        # smpl_data[:, 11] = smpl_data[:, 8] # R_Toe = R_Ankle
        # smpl_data[:, 10] = smpl_data[:, 7] + np.array([0, 0, -0.05])  # L_Toe 略微向下
        # smpl_data[:, 11] = smpl_data[:, 8] + np.array([0, 0, -0.05])  # R_Toe 略微向下
        smpl_data[:, 10] = 0.0      # L_Toe 标记为缺失
        smpl_data[:, 11] = 0.0      # R_Toe 标记为缺失
        
        
        # Arms
        # Collars (锁骨) 插值: 在 Spine3 和 Shoulder 之间
        if h36m_data[:, 8].all() == 0.0 or h36m_data[:, 14].all() == 0.0:
            smpl_data[:, 13] = 0.0
        else:
            smpl_data[:, 13] = h36m_data[:, 8] * 0.75 + h36m_data[:, 14] * 0.25 # L_Collar
        
        if h36m_data[:, 8].all() == 0.0 or h36m_data[:, 11].all() == 0.0:
            smpl_data[:, 14] = 0.0
        else:
            smpl_data[:, 14] = h36m_data[:, 8] * 0.75 + h36m_data[:, 11] * 0.25 # R_Collar
        
        smpl_data[:, 16] = h36m_data[:, 14] # L_Shoulder
        smpl_data[:, 18] = h36m_data[:, 15] # L_Elbow
        smpl_data[:, 20] = h36m_data[:, 16] # L_Hand
        
        smpl_data[:, 17] = h36m_data[:, 11] # R_Shoulder
        smpl_data[:, 19] = h36m_data[:, 12] # R_Elbow
        smpl_data[:, 21] = h36m_data[:, 13] # R_Hand
        
        return smpl_data

    def smpl22_to_h36m(self, smpl_data, original_root=None):
        """
        将 SMPL (22点, Y-Up) -> Human3.6M (17点, Y-Down)
        
        参数:
            smpl_data: (Frames, 22, 3)
            original_root: (Frames, 1, 3) 可选，用于将生成的骨骼还原回原始绝对位置
        """
        data = smpl_data.copy()
        frames = data.shape[0]
        h36m = np.zeros((frames, 17, 3), dtype=data.dtype)
        
        # 1. 关节点还原
        h36m[:, 0] = data[:, 0]   # Pelvis
        h36m[:, 1] = data[:, 1]   # L_Hip
        h36m[:, 2] = data[:, 4]   # L_Knee
        h36m[:, 3] = data[:, 7]   # L_Foot
        
        h36m[:, 4] = data[:, 2]   # R_Hip
        h36m[:, 5] = data[:, 5]   # R_Knee
        h36m[:, 6] = data[:, 8]   # R_Foot
        
        h36m[:, 7] = data[:, 3]   # Center Torso
        h36m[:, 8] = data[:, 9]   # Upper Torso
        h36m[:, 9] = data[:, 12]  # Neck
        h36m[:, 10] = data[:, 15] # Head
        
        h36m[:, 11] = data[:, 17] # R_Shoulder
        h36m[:, 12] = data[:, 19] # R_Elbow
        h36m[:, 13] = data[:, 21] # R_Hand
        
        h36m[:, 14] = data[:, 16] # L_Shoulder
        h36m[:, 15] = data[:, 18] # L_Elbow
        h36m[:, 16] = data[:, 20] # L_Hand
        
        #* 2. 坐标系还原 (Y-Flip)
        h36m[:, :, 1] *= -1
        
        # 3. 绝对位置还原 (如果有原始Root信息)
        if original_root is not None:
            h36m = h36m + original_root
            
        return h36m
import sys
import os
from pathlib import Path
import numpy as np
import torch

# --- 1. 路径设置 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(str(ROOT))

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def load_dataset_helper(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data

def main(prepare_kpts:np.ndarray=None, prepare_filter:np.ndarray=None, PrepareData_input_dirpath:str=None, results_output_dirpath:str=None,
         visualize_flag:bool=False):
    """
    该函数执行 SARS 动作补全任务，使用 OmniControl 模型根据部分观测点和文本提示生成完整的骨骼动作序列。
    
    :param prepare_kpts: Prepared keypoints array of shape (N, Frames, 22, 3), defaults to None
    :param prepare_filter: Prepared filter array of shape (N, 1, 1, Frames), defaults to None
    :param PrepareData_input_dirpath: Directory path of prepared input data, defaults to None
    :param results_output_dirpath: Directory path to save results, defaults to None
    """
    # --- 0. 前置检查&数据备份 ---
    if PrepareData_input_dirpath is not None and not os.path.exists(PrepareData_input_dirpath):
        raise FileNotFoundError(f"PrepareData_input_dirpath does not exist: {PrepareData_input_dirpath}")
    if results_output_dirpath is not None and not os.path.exists(results_output_dirpath):
        os.makedirs(results_output_dirpath, exist_ok=True)
    
    
    # [优化修改]：仅复制轻量的 .txt 文件用于查阅，彻底停止复制庞大且易损坏的 .pkl 和 .npy 输入数据
    # # Copy .npy, .pkl, .txt files from prepare directory to results directory
    # if PrepareData_input_dirpath:
    #     for root, dirs, files in os.walk(PrepareData_input_dirpath):
    #         for file in files:
    #             if file.endswith('.npy') or file.endswith('.pkl') or file.endswith('.txt'):
    #                 src = os.path.join(root, file)
    #                 dst = os.path.join(results_output_dirpath, 
    #                                   os.path.relpath(src, PrepareData_input_dirpath))
    #                 os.makedirs(os.path.dirname(dst), exist_ok=True)
    #                 shutil.copy2(src, dst)
    
    
    # --- A. 初始化 ---
    args = generate_args()
    fixseed(args.seed)
    
    vis_output_dir = pjoin(results_output_dirpath, 'visualize') if results_output_dirpath is not None else pjoin(ROOT, 'results', 'sars_imputation_results', 'visualize')
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir, exist_ok=True)
    print(f"Visualization results will be saved to: {vis_output_dir}")
    
    # 获取骨骼运动链定义 (用于画图连线)
    # OmniControl 基于 HumanML3D，所以使用 t2m_kinematic_chain
    kinematic_chain = paramUtil.t2m_kinematic_chain
    
    # 定义你的生成配置
    NUM_REPETITIONS = 10  # 每种动作生成 10 份 (作为一个 Batch 处理)
    args.batch_size = NUM_REPETITIONS # 显存允许的话，直接设为 10
    
    args.model_path = "./save/omnicontrol_ckpt/model_humanml3d.pt"
    args.dataset = 'humanml'
    args.num_samples = 1
    
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    n_frames = 196 # 你的数据帧数
    
    # 定义 5 种动作的文本提示 (请根据你的需求修改)
    TARGET_ACTIONS = [
        # "a person is moving forward",
        # "a person is carrying a piece of wood while walking",
        # "a person is climbing up on a ladder",
        # "a person is working for roof on a ladder",
        # "a person is working for measuring and cutting wood using a slide saw at a table"
        
        "a person is walking forward normally",                                                                                             # move
        "a person is walking forward while holding a heavy object in front of their chest with both hands",                                 # carry
        "a person is climbing up a vertical ladder, moving their arms and legs upwards alternately",                                    # climb
        "a person is standing still with their legs apart, reaching both arms upwards to interact with something above their head",     # Ladder_work
        "a person is standing in place, bending slightly forward, and moving their hands back and forth in front of them",              # Table_work
        
    ]
    NUM_ACTIONS = len(TARGET_ACTIONS)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # --- B1. 加载数据集元数据 ---
    print('Loading dataset wrapper...')
    data = load_dataset_helper(args, max_frames, n_frames)
    
    
    #* 注意：如果 prepare_kpts 已经是 (N, Frames, 22, 3) 的格式，并且坐标系已经是 Y-Up，那么这里就不需要再进行适配，可以直接使用 prepare_kpts 进行后续处理。
    #* 如果 prepare_kpts 是基于 Human3.6M 的 (N, Frames, 17, 3)，并且是 Y-Down 坐标系，那么你需要先使用 SkeletonAdapter 将其转换为 SMPL 的 (N, Frames, 22, 3) 格式，并且转换坐标系。同时需要进行归一化处理，确保数值范围与模型训练时一致。
    # --- B2. 加载归一化统计量 (Normalization Stats) ---
    print("Loading normalization statistics...")
    spatial_norm_path = './dataset/humanml_spatial_norm'
    # 确保文件存在，路径需根据项目结构调整
    mean_path = pjoin(spatial_norm_path, 'Mean_raw.npy')
    std_path = pjoin(spatial_norm_path, 'Std_raw.npy')
    assert os.path.exists(mean_path), f"Mean_raw.npy not found at {mean_path}"
    assert os.path.exists(std_path), f"Std_raw.npy not found at {std_path}"
    # 加载并转换为 Tensor
    raw_mean_np = np.load(mean_path)
    raw_std_np = np.load(std_path)
    # 形状 (22, 3) -> 压平 -> (1, 1, 66)
    raw_mean = torch.from_numpy(raw_mean_np.reshape(1, 1, -1)).float().to(device)
    raw_std = torch.from_numpy(raw_std_np.reshape(1, 1, -1)).float().to(device)
    
    

    # --- C. 加载模型 ---
    print("Creating model...")
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()

    # --- D. 数据准备 ---
    print("Preparing input data...")
    
    # 判断 prepare_kpts 非0值点，并构造对应的掩码 (Mask)，告诉模型哪些点是有效的，哪些点是缺失的
    # 生成 spatial_filter: (N, Frames, 22)，非0位置为 1.0，0 位置为 0.0
    prepareKpts_validMask = (np.linalg.norm(prepare_kpts, axis=-1) > 1e-6).astype(np.float32)  # (B, Frames, 22)
    # 调整 prepare_kpts 所有有效点的数值范围，使其与模型训练时的坐标范围一致(根据示例代码来看，所有数值都为正数，且范围大约在 0-2 之间，符合 SMPL 的坐标范围)
    prepare_kpts[..., 1] = prepare_kpts[..., 1] + prepareKpts_validMask * raw_mean[0, 0, 1].item()  #* 将所有有效节点的 Y 轴的数值整体上移 0.9，使其范围更接近 SMPL 的坐标范围 (根据你的数据分布调整这个偏移量)
    
    
    # prepare_kpts 输入形状为 (B, Frames, 22, 3) -> 需要调整为 (B, Frames, 66)
    prepare_kpts = prepare_kpts.reshape(prepare_kpts.shape[0], prepare_kpts.shape[1], -1)  # (B, Frames, 22, 3) -> (B, Frames, 66)
    prepare_filter = prepare_filter  # (B, 1, 1, Frames)
    
    assert prepare_kpts.shape[1] == n_frames, f"Input keypoints frames {prepare_kpts.shape[1]} does not match expected {n_frames}"
    assert prepare_filter.shape[-1] == n_frames, f"Input mask frames {prepare_filter.shape[-1]} does not match expected {n_frames}"
    
    # 结果容器: (Samples, Actions, Reps, Frames, 22, 3)
    final_results_container = []

    # --- E. 开始循环生成 ---
    # 外层循环：遍历每一个样本
    for sample_idx in range(prepare_kpts.shape[0]):
        print(f"Processing sample {sample_idx+1}/{prepare_kpts.shape[0]}...")
        
        # 获取当前样本的 hint 和 mask
        # current_hint: (Frames, 66)
        # current_mask: (1, 1, Frames)
        target_hint_np = prepare_kpts[sample_idx]
        target_filter_np = prepare_filter[sample_idx]
        
        sample_action_results = [] # 存放该样本的 5 种动作结果
        
        #* 内层循环：遍历 5 种动作
        for action_idx, action_text in enumerate(TARGET_ACTIONS):
            print(f"  > Generating Action: '{action_text}' ({NUM_REPETITIONS} reps)...")
            
            # --- 1. 构造 Hint Batch (Replication) ---
            # 我们将单个样本复制 10 份，组成一个 Batch
            
            # 1. Hint: (10, Frames, 66)
            # 调整为模型需要的 (Batch, Frames, Feats) -> 这里的 66 就是 Feats
            # 注意: 之前我们确认了不需要 permute，直接是 (B, T, D) 即可
            batch_hint_np = np.tile(target_hint_np[np.newaxis, ...], (NUM_REPETITIONS, 1, 1))
            batch_hint = torch.from_numpy(batch_hint_np).float().to(device)
            
            #! 关键：需要后续讨论归一化处理，反归一化处理对生成结果的影响，目前先按照之前的方式进行归一化处理，确保数值范围与模型训练时一致，这样模型才能正确理解输入的 Hint 信息
            #* --- 2. 执行数据归一化 ---
            # A. 创建空间掩码 (Spatial Hint Filter)：重新筛选并标记获取的数据中哪些点是有效的（非0）
            #    注意：原始数据中 0.0 代表缺失．
            #    形状: (10, Frames, 66): 非0位置为 1.0，0 位置为 0.0
            batch_spatial_hintFilter = (batch_hint.abs() > 1e-6).float()
            
            # B. 应用归一化公式: (X - Mean) / Std
            #    HumanML3D 的 Mean/Std 是基于 Y-up 坐标系的，请确保你的 prepare_kpts 已经是 Y-up
            batch_hint_norm = (batch_hint - raw_mean) / raw_std
            
            # C. 重新应用掩码：将原本是 0 的地方（现在变成了 -Mean/Std）强制置回 0
            #    这是最关键的一步，告诉模型这些点是缺失的
            batch_hint_norm = batch_hint_norm * batch_spatial_hintFilter
            # 现在 batch_hint_final 就是最终的归一化 Hint 输入
            # ------------------------------------------------------
            
            #* --- 3. 构造 Mask Batch (Temporal/Length Mask) ---
            # OmniControl 通常使用这个 mask 来指示序列长度，mask 由 lengths_to_mask 生成，用于标记序列实际长度。它的形状为 [batch, 1, 1, max_len]，可用于广播，告诉模型哪些帧有效。在 sample/generate.py 中，mask 被用在 rot2xyz 时构造时间步掩码（非 xyz 表示时）
            # mask 就是 “有效帧掩码”，表示哪些时间步是真实数据、哪些是 padding
            # 既然是定长生成 (196帧)，这里可以用全 True，或者沿用传入的 mask
            batch_temporal_filter_np = np.tile(target_filter_np[np.newaxis, ...], (NUM_REPETITIONS, 1, 1, 1))
            batch_temporal_filter = torch.from_numpy(batch_temporal_filter_np).bool().to(device)      # 转为 Bool
            
            # 3. Text: 10 个相同的 Prompt
            batch_texts = [action_text] * NUM_REPETITIONS
            
            # 4. Lengths
            batch_lengths = torch.tensor([n_frames] * NUM_REPETITIONS).to(device)

            # --- 构造输入字典 ---
            model_kwargs = {
                'y': {
                    'text': batch_texts,
                    'lengths': batch_lengths,
                    'hint': batch_hint_norm,     # (10, Frames, 66)
                    'mask': batch_temporal_filter,     # (10, 1, 1, Frames)
                    'scale': torch.ones(NUM_REPETITIONS, device=device) * args.guidance_param
                }
            }
            
            # --- 执行生成 ---
            # 一次生成 10 个 Repetitions
            sample = diffusion.p_sample_loop(
                model,
                (NUM_REPETITIONS, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                progress=False, # 关闭内部进度条以免刷屏
                const_noise=False, # 关键：关闭固定噪声，确保随机噪声不同，这样10个结果才会有差异
            )
            
            # --- 后处理 (Batch Processing) ---
            # 1. 提取有效特征 & 反归一化
            sample = sample[:, :263, ...]        # (10, 263, 1, 196). why 263? 因为 HumanML3D 有 263 个有效特征, 22个关节 * 3维 + 其他无效特征被丢弃(防止模型输出了多余的特征），如果你的模型输出已经是 66 维，那么这里就不需要切片了，直接使用 sample 就好
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()   # (10, 263, 1, 196) -> (10, 1, 196, 263)，这里的 inv_transform 会自动根据训练时的归一化方式进行反归一化处理，确保数值范围回到原始坐标系下
            
            # 2. 恢复 XYZ, 为什么要恢复 XYZ? 因为模型输出的是旋转不变的坐标，需要转换回全局坐标系下的关节位置
            n_joints = 22 if sample.shape[-1] == 263 else 21
            sample = recover_from_ric(sample, n_joints) # (10, 1, 196, 263) -> (10, 1, Frames, 22, 3)
            
            # 3. 调整维度 (10, 1, Frames, 22, 3) -> (10, Frames, 22, 3)
            sample = sample.squeeze(1) # (10, Frames, 22, 3)
            print(f"    Generated sample shape: {sample.shape}")
            sample = sample[:, :n_frames, :, :] # 截断到 n_frames, why? 以防万一模型输出多帧 (10, Frames, 22, 3)，但理论上不会
            
            # 4. 存入临时列表
            # sample shape: (10, Frames, 22, 3)
            sample_action_results.append(sample.cpu().numpy())
            
            
            # =======================================================
            # 5. 可视化模块 (只生成 Batch 中的第 0 个样本)
            # =======================================================
            if visualize_flag:
                try:
                    # A. 提取第 0 个生成的动作 (作为代表)
                    # shape: (Frames, 22, 3)
                    vis_motion = sample[0].cpu().numpy()
                    
                    # B. 准备 Hint 数据 (用于在图中画出红色的观测点)
                    # current_hint_np 是 (Frames, 66)，需要变回骨骼结构 (Frames, 22, 3)
                    vis_hint = target_hint_np.reshape(n_frames, 22, 3)
                    
                    # C. 构造文件名
                    # 格式: sample_01_action_01_climbing.mp4
                    save_title = action_text.replace(" ", "_").replace(".", "")[:20]
                    save_filename = f"sample_{sample_idx:02d}_action_{action_idx:02d}_{save_title}.mp4"
                    save_path = os.path.join(vis_output_dir, save_filename)
                    
                    # D. 调用 OmniControl 的绘图函数
                    # 注意: title 是显示在视频顶部的文字，fps 建议设为 20
                    print(f"    -> Saving visualization to {save_path} ...")
                    # plot_3d_motion(
                    #     save_path, 
                    #     kinematic_chain, 
                    #     vis_motion, 
                    #     dataset=args.dataset, 
                    #     title=action_text, 
                    #     fps=20, 
                    #     hint=vis_hint 
                    # )
                    plot_compare_3d_motion(
                        save_path,
                        kinematic_chain,
                        vis_motion,
                        dataset=args.dataset,
                        title=action_text,
                        fps=20,
                        hint=vis_hint
                    )
                    
                except Exception as e:
                    print(f"    [Warning] Visualization failed: {e}")
    
        # --- 保存该样本的所有动作结果 ---
        # 将该样本的所有动作结果堆叠
        # shape: (Motion_Categories, Motion_repetitions, Frames, 22, 3)
        sample_action_results = np.stack(sample_action_results, axis=0)
        final_results_container.append(sample_action_results)
    
    
    # --- F. 保存最终大文件 ---
    # 最终形状: (Num_Samples, Num_Actions, Num_Reps, Frames, 22, 3)
    final_results = np.stack(final_results_container, axis=0)
    
    print(f"\nAll Done!")
    print(f"Final Data Shape: {final_results.shape}")
    # 例如: (2, 5, 10, 64, 22, 3)
    
    results_output_abspath = osp.join(results_output_dirpath, "sars_imputation_results.npy") if results_output_dirpath is not None else "./results/sars_imputation_results.npy"
    np.save(results_output_abspath, final_results)
    print(f"Results saved to: {results_output_abspath}")
    
    # 额外保存一下动作列表，方便后续对应
    with open(osp.join(results_output_dirpath, "sars_imputation_action_list.txt"), "w") as f:
        for act in TARGET_ACTIONS:
            f.write(act + "\n")
    
    ...
    

    # # --- H. 还原回 H36M 并保存 ---
    # restored_list = []
    # sample_np = sample.cpu().numpy()
    
    # for i in range(args.batch_size):
    #     # 现在 sample_np[i] 是 (Frames, 22, 3)，完全符合适配器要求
    #     h36m_res = adapter.smpl22_to_h36m(sample_np[i])
    #     restored_list.append(h36m_res)
        
    # final_result = np.array(restored_list)
    # output_path = osp.join(osp.dirname(__file__), '..', 'save', "inpainted_results.npy")
    # np.save(output_path, final_result)
    # print(f"Done. Result shape: {final_result.shape}. Saved to {output_path}")

# if __name__ == "__main__":
#     # --- 强制注入命令行参数 ---
#     # 如果检测到没有输入参数（直接点击运行时），自动填入默认参数
#     import sys
#     import re
#     if len(sys.argv) == 1: 
#         print("Detected direct run. Injecting default arguments...")
#         sys.argv.extend([
#             "--model_path", "./save/omnicontrol_ckpt/model_humanml3d.pt",
#             # 如果还有其他必须参数，也可以在这里继续添加
#             # "--text_prompt", "A person is walking" 
#         ])
    
#     if sys.platform == 'win32' or sys.platform == 'cygwin':
#         HHD_ROOT = "K:\\"
#     elif sys.platform == 'linux':
#         HHD_ROOT = f"/media/{USER_NAME}/HHD_K/"
    
    
#     Prepare_target_dirname = "CustomDataset_20241214_GP_202603191518_noThred"   # 这个目录应该包含 upper/bottom/left/right/left_hand/right_hand/left_leg/right_leg/inter/norm 这几个子目录，每个子目录里有对应的 _kpts.npy 和 _mask.npy 文件
#     Output_target_dirname = Prepare_target_dirname + "_GG_{}".format(time.strftime('%Y%m%d%H%M', time.localtime()))
    
    
    
#     mask_cat = ['upper', 'bottom', 'left', 'right', 'left_hand', 'right_hand', 'left_leg', 'right_leg', 'inter', 'norm']
#     target_maskList = []
#     for mask_id in target_maskList if len(target_maskList) > 0 else mask_cat:
#         # if mask_id == 'norm':       continue
        
#         # --------------------------
#         PrepareData_input_dirpath = osp.join(HHD_ROOT, 'SARS-Inter_DL', 'G_Prepare', Prepare_target_dirname, mask_id)
#         Output_generated_dirpath = osp.join(HHD_ROOT, 'SARS-Inter_DL', 'G_Generate', Output_target_dirname, mask_id)
        
#         if not osp.exists(Output_generated_dirpath):
#             os.makedirs(Output_generated_dirpath, exist_ok=True)
        
#         prepare_kpts_path = [osp.join(PrepareData_input_dirpath, file) for file in os.listdir(PrepareData_input_dirpath) if re.search(f'\w*_kpts\.npy$', file)][-1]
#         prepare_mask_path = [osp.join(PrepareData_input_dirpath, file) for file in os.listdir(PrepareData_input_dirpath) if re.search(f'\w*_mask\.npy$', file)][-1]
        
#         assert osp.exists(prepare_kpts_path), f"prepare_kpts_path not found: {prepare_kpts_path}"
#         assert osp.exists(prepare_mask_path), f"prepare_mask_path not found: {prepare_mask_path}"
        
#         prepare_kpts = np.load(prepare_kpts_path)   # (N, 196, 22, 3)
#         prepare_mask = np.load(prepare_mask_path)   # (N, 1, 1, 22)
#         # --------------------------
    
#         main(prepare_kpts, prepare_mask, PrepareData_input_dirpath, Output_generated_dirpath)


if __name__ == "__main__":
    # --- 强制注入命令行参数 ---
    import sys
    import re
    import getpass  # [新增] 引入 getpass 以获取用户名
    
    if len(sys.argv) == 1: 
        print("Detected direct run. Injecting default arguments...")
        sys.argv.extend([
            "--model_path", "./save/omnicontrol_ckpt/model_humanml3d.pt",
        ])
    
    USER_NAME = getpass.getuser() # [新增] 获取用户名
    
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        HHD_ROOT = "K:\\"
    elif sys.platform == 'linux':
        HHD_ROOT = f"/media/{USER_NAME}/HHD_K/"
    
    
    visualize_flag = False   # [新增] 可视化控制开关：True 为开启画图，False 为关闭画图直接生成数据
    
    
    
    # [修改提示] 请将这里替换为 G_OmniContral_inputPrepare.py 生成的准备数据文件夹名称
    Prepare_target_dirname = "CustomDataset_20241214_GP_202603231227_TH50"      # 这个目录应该包含 upper/bottom/left/right/left_hand/right_hand/left_leg/right_leg/inter/norm 这几个子目录，每个子目录里有对应的 _kpts.npy 和 _mask.npy 文件
    Output_target_dirname = Prepare_target_dirname + "_GG_{}".format(time.strftime('%Y%m%d%H%M', time.localtime()))
    
    mask_cat = ['upper', 'bottom', 'left', 'right', 'left_hand', 'right_hand', 'left_leg', 'right_leg', 'inter', 'norm']
    target_maskList = []
    
    # [核心修改 1]：引入需要遍历的模型列表，与 G_Prepare 和后续的 Integrator 保持绝对一致
    recognize_model_list = ['stgcn', 'stgcnpp', 'ctrgcn', 'posec3d_kpts', 'posec3d_limbs', 'msg3d']
    
    for mask_id in target_maskList if len(target_maskList) > 0 else mask_cat:
        # if mask_id == 'norm':       continue
        
        # [核心修改 2]：增加对 recognize_model_list 的遍历
        for target_recog_model in recognize_model_list:
            print(f"\n=======================================================")
            print(f"🚀 Starting Generation | Mask: {mask_id} | Model: {target_recog_model.upper()}")
            print(f"=======================================================\n")
            
            # [核心修改 3]：路径深入到 target_recog_model 这一级子目录
            PrepareData_input_dirpath = osp.join(HHD_ROOT, 'SARS-Inter_DL', 'G_Prepare', Prepare_target_dirname, mask_id, target_recog_model)
            Output_generated_dirpath = osp.join(HHD_ROOT, 'SARS-Inter_DL', 'G_Generate', Output_target_dirname, mask_id, target_recog_model)
            
            # 安全校验：如果某个模型在某个 mask 下没有需要补全的数据（比如精度都很高被阈值过滤了），则跳过
            if not osp.exists(PrepareData_input_dirpath):
                print(f"  [Info] Input dir not found (No samples to generate), skip: {PrepareData_input_dirpath}")
                continue
            
            if not osp.exists(Output_generated_dirpath):
                os.makedirs(Output_generated_dirpath, exist_ok=True)
            
            # 使用 raw string (r'') 修复可能引发的正则表达式转义警告
            try:
                prepare_kpts_path = [osp.join(PrepareData_input_dirpath, file) for file in os.listdir(PrepareData_input_dirpath) if re.search(r'\w*_kpts\.npy$', file)][-1]
                prepare_mask_path = [osp.join(PrepareData_input_dirpath, file) for file in os.listdir(PrepareData_input_dirpath) if re.search(r'\w*_mask\.npy$', file)][-1]
            except IndexError:
                print(f"  [Warning] Missing .npy files in {PrepareData_input_dirpath}, skip.")
                continue
            
            assert osp.exists(prepare_kpts_path), f"prepare_kpts_path not found: {prepare_kpts_path}"
            assert osp.exists(prepare_mask_path), f"prepare_mask_path not found: {prepare_mask_path}"
            
            prepare_kpts = np.load(prepare_kpts_path)   # (N, 196, 22, 3)
            prepare_mask = np.load(prepare_mask_path)   # (N, 1, 1, 196) 
            
            # [防呆校验]：如果提取出来的数据量为 0，跳过生成器调用，防止底层 Pytorch 崩溃
            if prepare_kpts.shape[0] == 0:
                print(f"  [Info] 0 samples to generate in {PrepareData_input_dirpath}, skip.")
                
                # [新增] 为了方便查阅，即使不需要生成数据，也创建文件夹并复制 sample_list.txt
                import shutil
                os.makedirs(Output_generated_dirpath, exist_ok=True)
                try:
                    # 寻找并复制 sample_list.txt
                    sample_list_file = [f for f in os.listdir(PrepareData_input_dirpath) if f.endswith('_sample_list.txt')][-1]
                    shutil.copy2(osp.join(PrepareData_input_dirpath, sample_list_file), 
                                 osp.join(Output_generated_dirpath, sample_list_file))
                    print(f"  [Info] Copied {sample_list_file} to Output directory for reference.")
                except IndexError:
                    pass
                
                continue
            
            # --------------------------
            # 正式调用 OmniControl 进行推理生成
            main(prepare_kpts, prepare_mask, PrepareData_input_dirpath, Output_generated_dirpath, visualize_flag=visualize_flag)