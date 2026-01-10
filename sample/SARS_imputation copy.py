# run_imputation.py
import argparse
import os
import numpy as np
import torch

# 引入 OmniControl 的必要模块
#以此确保脚本能找到 sample 模块 (如果报错，请确保你在项目根目录运行)
from sample.generate import load_model_from_args, get_args_per_step
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import generate_args

import pickle
import os.path as osp

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

def main():
    # 1. 基础参数设置 (继承官方的参数解析)
    args = generate_args()
    args.num_repetitions = 1 # 我们只需要为每个输入生成一个补全结果(或者你可以设为10来挑选)
    
    # 强制修改一些参数以适配你的任务
    args.model_path = "./save/omnicontrol_ckpt/model_humanml3d.pt" 
    # args.text_prompt = "" # 后面我们会根据具体动作动态设置
    
    # 初始化设备
    fixseed(args.seed)
    dist_util = None # 单卡运行不需要复杂的分布式设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载模型
    print(f"Loading model from {args.model_path}...")
    model, diffusion = create_model_and_diffusion(args, None)
    
    # 加载权重
    model_state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # 3. 准备你的数据 (模拟读取你的 .npy)
    # TODO: 替换为你真实的数据读取逻辑
    # 假设 raw_h36m_batch 是 (Batch, Frames, 17, 3)
    # 假设 raw_confidence 是 (Batch, Frames, 17)
    print("Preparing input data...")
    
    # --- [模拟数据开始] --- 
    # 这里我造一个假数据用于演示，实际使用时请替换为 np.load('your_data.npy')
    # 假设有 2 个样本，64帧
    batch_size = 2 
    num_frames = 64
    raw_h36m_datasetPKL = pickle.load(open(r'K:\\SARS-Inter_Data\\Recog_data\\CustomDataset_20241214\\upper\\h36m_prop_20.pkl', "rb"))  # (2, 64, 17, 3)
    
    def extract_test_data(datasetPKL):
        target_fileNameList = datasetPKL['split']['test']
        
        data_list = []
        for _anno in datasetPKL['annotations']:
            if _anno['fileName'] in target_fileNameList:
                data_list.append(_anno['keypoint'])
        
        assert len(data_list) == len(target_fileNameList), "数据集划分与读取不匹配，请检查"
        return np.array(data_list)
    
    raw_h36m_dataset = extract_test_data(raw_h36m_datasetPKL)  # (N, 64, 17, 3)
    # 模拟遮挡：假设下半身置信度为 0
    raw_confidence = np.ones((batch_size, num_frames, 17))
    raw_confidence[:, :, 1:7] = 0 # 遮挡下半身
    # --- [模拟数据结束] ---

    # 初始化适配器
    adapter = SkeletonAdapter()

    # 4. 数据转换与 Mask 构建
    # 容器
    hints_list = []
    masks_list = []
    lengths_list = []

    for i in range(batch_size):
        # A. 转换坐标: H36M(17) -> SMPL(22)
        # 注意: h36m_to_smpl22 接收 (Frames, 17, 3)
        smpl_data = adapter.h36m_to_smpl22(raw_h36m_batch[i]) # (64, 22, 3)
        
        # B. 转换 Confidence/Mask
        # 我们需要手动构建 SMPL 的 mask。逻辑是:
        # 如果 H36M 的源节点置信度 < 0.5，则对应的 SMPL 节点 Mask = 0 (缺失)
        h36m_conf = raw_confidence[i] > 0.5 # (64, 17) Boolean
        
        # 初始化 SMPL mask (64, 22)
        smpl_mask = np.ones((num_frames, 22), dtype=bool)
        
        # 映射逻辑 (参考 Adapter):
        # 下半身
        smpl_mask[:, 1] = h36m_conf[:, 1] # L_Hip
        smpl_mask[:, 4] = h36m_conf[:, 2] # L_Knee
        smpl_mask[:, 7] = h36m_conf[:, 3] # L_Ankle
        smpl_mask[:, 10] = False          # Toe 永远设为缺失，让模型生成
        
        smpl_mask[:, 2] = h36m_conf[:, 4] # R_Hip
        smpl_mask[:, 5] = h36m_conf[:, 5] # R_Knee
        smpl_mask[:, 8] = h36m_conf[:, 6] # R_Ankle
        smpl_mask[:, 11] = False          # Toe 永远设为缺失

        # 躯干
        smpl_mask[:, 0] = h36m_conf[:, 0] # Pelvis
        smpl_mask[:, 3] = h36m_conf[:, 7] # Spine1
        smpl_mask[:, 9] = h36m_conf[:, 8] # Spine3
        smpl_mask[:, 12] = h36m_conf[:, 9] # Neck
        smpl_mask[:, 15] = h36m_conf[:, 10] # Head
        
        # 插值点 (Spine2, Collars)
        # 策略: 如果两端都存在，则设为存在；否则设为缺失
        smpl_mask[:, 6] = h36m_conf[:, 7] & h36m_conf[:, 8] # Spine2
        smpl_mask[:, 13] = h36m_conf[:, 8] & h36m_conf[:, 14] # L_Collar
        smpl_mask[:, 14] = h36m_conf[:, 8] & h36m_conf[:, 11] # R_Collar
        
        # 手臂
        smpl_mask[:, 16] = h36m_conf[:, 14]
        smpl_mask[:, 18] = h36m_conf[:, 15]
        smpl_mask[:, 20] = h36m_conf[:, 16]
        
        smpl_mask[:, 17] = h36m_conf[:, 11]
        smpl_mask[:, 19] = h36m_conf[:, 12]
        smpl_mask[:, 21] = h36m_conf[:, 13]

        # C. 应用 Mask 到 Data (将缺失部分的坐标强制置0，虽然 OmniControl 内部可能也会处理，但这样更安全)
        # smpl_mask shape: (64, 22) -> 扩展到 (64, 22, 3)
        smpl_data[~smpl_mask] = 0 
        
        # D. 调整维度以适配 OmniControl 输入
        # OmniControl Expects: 
        # Hint: (Joints, 3, Frames)
        # Mask: (Joints, 1, Frames)
        
        hint_tensor = torch.from_numpy(smpl_data).float().permute(1, 2, 0) # (22, 3, 64)
        mask_tensor = torch.from_numpy(smpl_mask).float().unsqueeze(2).permute(1, 2, 0) # (22, 1, 64)
        
        hints_list.append(hint_tensor)
        masks_list.append(mask_tensor)
        lengths_list.append(num_frames)

    # 堆叠 Batch
    hint_batch = torch.stack(hints_list).to(device) # (Batch, 22, 3, 64)
    mask_batch = torch.stack(masks_list).to(device) # (Batch, 22, 1, 64)
    
    # 5. 设置 Text Prompt
    # 你可以根据动作分类给每个样本不同的 Prompt，或者用通用 Prompt
    # 这里我们演示用通用 Prompt
    text_prompts = ["A construction worker performing an action."] * batch_size

    # 6. 构造模型输入字典 (model_kwargs)
    model_kwargs = {
        'y': {
            'text': text_prompts,
            'lengths': lengths_list,
            # 'mask': mask_batch, # 注意: 这里的 mask 有时是指 text mask，Spatial mask 通常叫 'hint'
            # OmniControl 具体实现中，spatial condition 是通过 'hint' 传入的
            'hint': hint_batch,
            # 有些实现可能需要 mask 单独传入，或者包含在 hint 逻辑中
            # 对于 OmniControl，通常它需要一个 mask 来指示 hint 的有效性
            # 我们查看 OmniControl 源码，通常在 classifier_free_guidance 内部处理
            # 关键：我们需要把 mask 也放进 model_kwargs，或者在 forward 能够访问到的地方
        }
    }
    
    # *** 关键修正 ***
    # OmniControl 的 spatial condition 需要显式传入 mask
    # 如果 model forward 函数签名支持 mask，则直接传入
    model_kwargs['y']['mask'] = mask_batch # 这里的 key 可能需要根据源码微调，通常是 'mask' 或 'spatial_mask'

    print("Starting generation (Inpainting)...")
    
    # 7. 执行生成 (Sampling)
    # 使用 diffusion.p_sample_loop 进行生成
    sample_fn = diffusion.p_sample_loop
    
    # 形状: (Batch, 263, Frames) -> HumanML3D 的特征空间
    # OmniControl 会在内部根据 'hint' 进行引导
    sample = sample_fn(
        model,
        (batch_size, model.n_joints, 3, num_frames), # 这里 OmniControl 输出直接是 (B, 22, 3, T) 还是 (B, 263, T)?
        # HumanML3D 模型通常输出 263 维特征。但 OmniControl 这种空间控制模型，
        # 往往输出调整后的骨骼位置。我们需要确认 output shape。
        # 如果是标准 MDM，shape 是 (B, 263, 1, T)。如果是 OmniControl 专门优化的，可能是 (B, 22, 3, T)。
        # 根据 OmniControl 论文，它基于 MDM，所以输出很可能是 latent feature (263)。
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 表示从纯噪声开始生成
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    # 8. 结果处理
    # sample 的输出通常是 (Batch, 22, 3, Frames) 或者 (Batch, 263, Frames)
    # OmniControl 如果是 Spatial Control，通常会解码回关节位置
    print(f"Output shape: {sample.shape}")

    # 如果输出是 (Batch, 22, 3, Frames)
    sample = sample.permute(0, 3, 1, 2).cpu().numpy() # (Batch, Frames, 22, 3)

    # 还原回 H36M
    restored_h36m_list = []
    for i in range(batch_size):
        # 还原
        h36m_res = adapter.smpl22_to_h36m(sample[i])
        restored_h36m_list.append(h36m_res)
        
    restored_h36m = np.array(restored_h36m_list)
    print(f"Restored H36M shape: {restored_h36m.shape}")

    # 保存结果
    np.save(osp.join(osp.dirname(__file__), '..', 'save', "inpainted_results.npy"), restored_h36m)
    print("Done. Saved to inpainted_results.npy")

if __name__ == "__main__":
    main()