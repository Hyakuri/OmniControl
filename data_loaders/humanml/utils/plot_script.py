# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/shared/centos7/ffmpeg/20190305/bin/ffmpeg'
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

from matplotlib.animation import FuncAnimation, FFMpegWriter


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    dataset,
    figsize=(6, 6),
    fps=20,
    radius=3,
    vis_mode="default", gt_frames=[], hint=None,):
    
    
    matplotlib.use("Agg")
    title = "\n".join(wrap(title, 30))

    def init_axis():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 2, radius / 2])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(elev=120, azim=-90)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.35))
        ax.add_collection3d(xz_plane)
        return xz_plane


    # ---- figure & 3d axis ----
    fig = plt.figure(figsize=figsize, dpi=250)  # dpi 200~300 通常就很清晰
    plt.tight_layout()
    ax = fig.add_subplot(111, projection="3d")
    
    # dataset-specific scale
    if dataset == "kit":
        scale = 0.003
    else:
        # OmniControl 里传进来的 args.dataset 常是 humanml3d/humanml
        # 这里按原逻辑：humanml 缩放 1.3
        scale = 1.3
    # ---- hint normalize ----
    # 支持:
    #   (T, 3)         每帧一个点
    #   (T, K, 3)      每帧 K 个点
    #   (K, 3)         固定点
    hint_arr = None
    if hint is not None:
        hint_arr = np.asarray(hint).copy() * scale
    
    # ---- colors ----
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange
    if vis_mode == "upper_body":
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == "gt":
        colors = colors_blue
    
    # ---- joints: (T, J*3) or (T, J, 3) ----
    # (T, J, 3)
    data_world = joints.copy().reshape(len(joints), -1, 3)
    
    print("plotting motion with", data_world.shape[0], "frames and", data_world.shape[1], "joints.")
    print("joints:", joints.shape)

    # preparation related to specific datasets
    if dataset == 'kit':
        scale = 0.003
    elif dataset == 'humanml':
        scale = 1.3
    else:
        # humanml3d / humanml3d-like
        scale = 1.3

    data_world *= scale

    # hint: (T, 22, 3)
    hint_world = None
    if hint is not None:
        print("hint shape:", np.shape(hint))
        hint_world = np.asarray(hint).copy() * scale

    # stats for plane
    MINS = data_world.min(axis=0).min(axis=0)
    MAXS = data_world.max(axis=0).max(axis=0)
    frame_number = data_world.shape[0]

    # put on floor (world)
    height_offset = MINS[1]
    data_world[:, :, 1] -= height_offset
    if hint_world is not None:
        hint_world[..., 1] -= height_offset

    # root trajectory in world xz
    trajec = data_world[:, 0, [0, 2]].copy()  # (T,2)

    # local skeleton for centered character
    data_local = data_world.copy()
    data_local[..., 0] -= data_world[:, 0:1, 0]
    data_local[..., 2] -= data_world[:, 0:1, 2]


    hint_joint_ids = None
    if hint_world is not None and hint_world.ndim == 3 and hint_world.shape[-1] == 3:
        T, K, _ = hint_world.shape
        J = data_world.shape[1]
        hint_joint_ids = []
        for k in range(K):
            votes = []
            # 每隔几帧取样，降低计算量
            for t in range(0, T, max(1, T // 50)):
                p = hint_world[t, k]  # (3,)
                d = ((data_world[t] - p) ** 2).sum(axis=1)  # (J,)
                votes.append(int(d.argmin()))
            # 多数票
            hint_joint_ids.append(max(set(votes), key=votes.count))

    init_axis()

    def update(index):
        # 清理上一帧（同时清掉 texts）
        for a in list(ax.lines):
            a.remove()
        for c in list(ax.collections):
            c.remove()
        for t in list(ax.texts):
            t.remove()

        init_axis()  # 每帧重置相机/范围，避免被 collection 影响

        artists = []

        # 地面：跟随人物 root 位移移动（世界坐标减去当前 root）
        plane = plot_xzPlane(
            MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1],
        )
        artists.append(plane)

        # hint：也是世界坐标减去当前 root（与地面同步移动）
        if hint_world is not None:
            # 当前帧 22 个关节提示点 (22,3)
            pts = hint_world[index].copy()
            pts[:, 0] -= trajec[index, 0]
            pts[:, 2] -= trajec[index, 1]

            sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="red", s=18, depthshade=False)
            artists.append(sc)

            # 轨迹线：对每个关节画出轨迹
            # 注意：画 22 条轨迹会更耗时；你可以只画选定关节，见下面“性能优化”
            for j in range(pts.shape[0]):
                tr = hint_world[: , j, :].copy()  # (t,3)
                tr[:, 0] -= trajec[index, 0]
                tr[:, 2] -= trajec[index, 1]
                ln = ax.plot3D(tr[:, 0], tr[:, 1], tr[:, 2], color="#80B79A", linewidth=1.6, alpha=0.9)[0]
                artists.append(ln)

                # 关节归属标注：第 j 条轨迹就是关节 j
                # 只在当前点旁边标一次即可（避免太乱）
                txt = ax.text(pts[j, 0], pts[j, 1], pts[j, 2], f"{j}",
                            fontsize=7, color="#2F6F5F")
                artists.append(txt)

        # 骨架
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ln = ax.plot3D(
                data_local[index, chain, 0],
                data_local[index, chain, 1],
                data_local[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )[0]
            artists.append(ln)

        # 关节编号
        # 注意：全标注会更慢，但 22 个一般还能接受
        for j in range(data_local.shape[1]):
            x, y, z = data_local[index, j, 0], data_local[index, j, 1], data_local[index, j, 2]
            txt = ax.text(x, y, z, str(j), color="black", fontsize=7)
            artists.append(txt)

        return artists

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps, bitrate=4000)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    update(0)
    plt.savefig(save_path.replace(".mp4", "_debug.png"), dpi=2000)

    plt.close()



def plot_compare_3d_motion(
    save_path,
    kinematic_tree,
    joints,
    title,
    dataset,
    figsize=(15, 6), # [修改] 增加画布宽度以容纳双图
    fps=20,
    radius=3,
    vis_mode="default",
    gt_frames=[],
    hint=None,
):
    matplotlib.use("Agg")
    title = "\n".join(wrap(title, 30))

    # 判断节点是否有效 (三轴绝对值之和大于 1e-6)
    def is_valid_joint(joint_coord):
        return np.sum(np.abs(joint_coord)) > 1e-6

    # 动态初始化 Axis 的辅助函数
    def init_axis(ax, title_str, show_grid=False):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_title(title_str, fontsize=9)
        
        if show_grid:
            ax.grid(True) # [修改] Hint图开启网格以便观察坐标
            ax.set_axis_on()
            # 设置刻度以便观察数值范围
            ax.set_xlabel('X')
            ax.set_ylabel('Y (Height)')
            ax.set_zlabel('Z')
        else:
            ax.grid(False)
            ax.set_axis_off()
            
        ax.view_init(elev=120, azim=-90)

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.35))
        ax.add_collection3d(xz_plane)
        return xz_plane

    # [修改] 创建双子图布局
    if hint is not None:
        fig = plt.figure(figsize=figsize, dpi=150)
        ax1 = fig.add_subplot(131, projection="3d") # 左图：生成结果
        ax2 = fig.add_subplot(132, projection="3d") # 中图：生成结果（带原始输入提示）
        ax3 = fig.add_subplot(133, projection="3d") # 右图：原始输入
        
        # # 调整子图位置
        # ax1.set_position([0.0, 0.1, 0.3, 0.8]) 
        # ax2.set_position([0.35, 0.1, 0.3, 0.8])
        # ax3.set_position([0.7, 0.1, 0.3, 0.8])
        plt.subplots_adjust(wspace=0.1) # 减小子图间距
    else:
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax1 = fig.add_subplot(111, projection="3d")
        ax2 = None
        ax3 = None

    # --- 1. 处理生成数据 (OmniControl Result) ---
    if dataset == "kit": scale = 0.003
    else:                scale = 1.3 # HumanML3D

    data_world = joints.copy().reshape(len(joints), -1, 3)
    data_world *= scale

    MINS = data_world.min(axis=0).min(axis=0)
    MAXS = data_world.max(axis=0).max(axis=0)
    frame_number = data_world.shape[0]

    # 地面高度校准 (Generated)
    height_offset = MINS[1]
    data_world[:, :, 1] -= height_offset
    
    # 轨迹计算 (Generated)
    trajec = data_world[:, 0, [0, 2]].copy()
    
    # 局部坐标 (Generated)
    data_local = data_world.copy()
    data_local[..., 0] -= data_world[:, 0:1, 0]
    data_local[..., 2] -= data_world[:, 0:1, 2]

    # --- 2. 处理 Hint 数据 (Original Input) ---
    hint_local = None
    hint_trajec = None
    hint_MINS = None
    hint_MAXS = None
    
    if hint is not None and ax3 is not None:
        # [关键] 独立处理 Hint 数据，不依赖生成数据的轨迹
        hint_world = np.asarray(hint).copy().reshape(len(hint), -1, 3)
        hint_world *= scale # 同样应用缩放
        
        # [优化] 计算 Hint 范围时，排除无效节点（坐标全为0的节点），避免极端值影响地面高度和范围计算
        # valid_hint_mask = np.sum(np.abs(hint_world), axis=-1) > 1e-6
        # if np.any(valid_hint_mask):
        #     # 获取所有有效点的坐标
        #     # 注意: valid_pts 变成了一个二维数组 (N_valid, 3)，其中 N_valid 是所有帧中有效节点的总数
        #     valid_pts = hint_world[valid_hint_mask]
            
        #     # [修复]: 二维数组只需要调用一次 min/max 即可得到 (3,) 的 x,y,z 极值
        #     hint_MINS = valid_pts.min(axis=0)
        #     hint_MAXS = valid_pts.max(axis=0)
        # else:
        #     # 如果没有有效点，使用生成数据的范围作为 fallback
        #     hint_MINS = MINS.copy()
        #     hint_MAXS = MAXS.copy()

        #* [修改] 直接使用生成数据的范围作为 Hint 的范围，确保地面高度一致，便于对比观察
        hint_MINS = MINS.copy()
        hint_MAXS = MAXS.copy()
        
        # hint_MINS = hint_world.min(axis=0).min(axis=0)
        # hint_MAXS = hint_world.max(axis=0).max(axis=0)
        
        # [关键] 独立计算 Hint 的地面高度
        # 这样即使原始数据是"贴地"的(Y值偏小)，这里也会强制把它抬升到 0 平面以上显示
        # 从而方便观察骨架结构
        hint_height_offset = hint_MINS[1]
        
        #* [关键] 只对有效节点应用高度偏移，无效节点保持为 0
        for f in range(hint_world.shape[0]):
            for j in range(hint_world.shape[1]):
                if is_valid_joint(hint_world[f, j]):
                    hint_world[f, j, 1] -= hint_height_offset
        
        # 独立轨迹
        hint_trajec = hint_world[:, 0, [0, 2]].copy()
        
        # 独立局部坐标
        hint_local = hint_world.copy()
        for f in range(hint_world.shape[0]):
            for j in range(hint_world.shape[1]):
                if is_valid_joint(hint_world[f, j]):
                    hint_local[f, j, 0] -= hint_world[f, 0, 0]
                    hint_local[f, j, 2] -= hint_world[f, 0, 2]
        # hint_local[..., 0] -= hint_world[:, 0:1, 0]
        # hint_local[..., 2] -= hint_world[:, 0:1, 2]

    # 颜色定义
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange

    # --- 动画更新函数 ---
    def update(index):
        artists = []
        
        # ==========================================
        # 解析当前帧的 Hint 缺失状态
        # ==========================================
        missing_joints = []
        valid_joints = []
        if hint_world is not None:
            pts_current = hint_world[index]
            for j in range(pts_current.shape[0]):
                if is_valid_joint(pts_current[j]):
                    valid_joints.append(str(j))
                else:
                    missing_joints.append(str(j))
                    
        miss_str = ",".join(missing_joints) if missing_joints else "None"
        valid_str = ",".join(valid_joints) if valid_joints else "None"
        
        # ===========================
        # 1. 左图：纯生成 (Generated)
        # ===========================
        ax1.clear()
        init_axis(ax1, f"Generated Result\n{title}", show_grid=True) # 开启网格以便观察坐标
        
        # 画地面 (左), 地面跟随生成的 Trajectory 移动
        plot_xzPlane(
            ax1,
            MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1],
        )

        # 画骨骼 (左 - 橙色系)
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax1.plot3D(
                data_local[index, chain, 0],
                data_local[index, chain, 1],
                data_local[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )
        
        # 关节编号 (左图)
        for j in range(data_local.shape[1]):
            x, y, z = data_local[index, j, 0], data_local[index, j, 1], data_local[index, j, 2]
            txt = ax1.text(x, y, z, str(j), color="black", fontsize=7)
            artists.append(txt)
        
        # ===========================
        # 2. 中图：对比结合 (Combined: Generated + Original Hint)
        # ===========================
        if ax2 is not None and hint_world is not None:
            ax2.clear()
            # [新增] 在中图标题添加缺失信息
            combined_title = f"Combined (Gen + Hint)\nMissing: [{miss_str}]"
            
            init_axis(ax2, combined_title, show_grid=True) # 开启网格以便观察坐标

            # 地面：跟随人物 root 位移移动（世界坐标减去当前 root）
            plane = plot_xzPlane(
                ax2,
                MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
                0,
                MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1],
            )
            artists.append(plane)

            # --- 画 Hint (红色，带轨迹) ---
            # 逻辑：Hint 的世界坐标 - 生成的根节点坐标 = 相对生成人物的位置
            # 这样如果生成的动作走偏了，你会看到红点和人分离；如果生成的动作正确，红点会紧贴在对应的关节上
            # hint：也是世界坐标减去当前 root（与地面同步移动）
            # 当前帧 22 个关节提示点 (22,3)
            pts = hint_world[index].copy()
            # [新增] 定义两种相近的轨迹颜色，用于交替显示（例如：薄荷绿 和 湖蓝色）
            traj_colors = ["#80B79A", "#34C1E2"]
            
            for j in range(pts.shape[0]):
                # ==========================================
                # 1. 无论当前帧是否可见，都绘制【历史轨迹】(从0帧到当前帧)
                # ==========================================
                tr_history = hint_world[:index+1, j, :].copy()
                valid_history_mask = np.sum(np.abs(tr_history), axis=-1) > 1e-6
                
                if np.any(valid_history_mask):
                    # 转换到相对生成 root 的坐标系
                    tr_history[:, 0] -= trajec[index, 0]
                    tr_history[:, 2] -= trajec[index, 1]
                    
                    # 寻找连续的有效段 (通过判断有效索引的差值是否大于1)
                    valid_indices = np.where(valid_history_mask)[0]
                    split_positions = np.where(np.diff(valid_indices) > 1)[0] + 1
                    segments = np.split(valid_indices, split_positions)
                    
                    # 遍历每一个连续段，交替使用不同颜色绘制
                    for seg_idx, seg in enumerate(segments):
                        if len(seg) > 1: # 至少需要两个点才能连成线
                            seg_coords = tr_history[seg]
                            c = traj_colors[seg_idx % len(traj_colors)]
                            ln = ax2.plot3D(seg_coords[:, 0], seg_coords[:, 1], seg_coords[:, 2], 
                                            color=c, linewidth=1.6, alpha=0.9)[0]
                            artists.append(ln)
                
                if is_valid_joint(pts[j]):
                    # 调整坐标到相对位置（相对于生成的根节点）
                    pts[j, 0] -= trajec[index, 0]
                    pts[j, 2] -= trajec[index, 1]

                    # 画散点
                    sc = ax2.scatter(pts[j, 0], pts[j, 1], pts[j, 2], color="red", s=18, depthshade=False)
                    artists.append(sc)
                    
                    # # 画轨迹线 (只画有效帧的轨迹)
                    # tr = hint_world[:, j, :].copy()
                    # # 过滤轨迹线中的无效点，避免画出连向原点的线
                    # valid_tr_mask = np.sum(np.abs(tr), axis=-1) > 1e-6
                    # tr_valid = tr[valid_tr_mask]
                    # if len(tr_valid) > 0:
                    #     tr_valid[:, 0] -= trajec[index, 0]
                    #     tr_valid[:, 2] -= trajec[index, 1]
                    #     ln = ax2.plot3D(tr_valid[:, 0], tr_valid[:, 1], tr_valid[:, 2], color="#80B79A", linewidth=1.6, alpha=0.9)[0]
                    #     artists.append(ln)
                    
                    
                    # 关节归属标注：第 j 条轨迹就是关节 j
                    # 只在当前点旁边标一次即可（避免太乱）
                    txt = ax2.text(pts[j, 0], pts[j, 1], pts[j, 2], f"{j}", fontsize=7, color="#2F6F5F")
                    artists.append(txt)
            # ----------------------------


            # 画骨骼 (中 - 橙色系)
            used_colors = colors_blue if index in gt_frames else colors
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                linewidth = 4.0 if i < 5 else 2.0
                ln = ax2.plot3D(
                    data_local[index, chain, 0],
                    data_local[index, chain, 1],
                    data_local[index, chain, 2],
                    linewidth=linewidth,
                    color=color,
                )[0]
                artists.append(ln)

            # 关节编号
            # 注意：全标注会更慢，但 22 个一般还能接受
            for j in range(data_local.shape[1]):
                x, y, z = data_local[index, j, 0], data_local[index, j, 1], data_local[index, j, 2]
                txt = ax2.text(x, y, z, str(j), color="black", fontsize=7)
                artists.append(txt)


        # ===========================
        # 3. 右图：纯原始 (Original Hint)
        # ===========================
        if ax3 is not None and hint_local is not None:
            ax3.clear()
            # [新增] 在右图标题添加详细的有效和缺失信息
            hint_title = f"Original Hint\nValid: [{valid_str}]\nMissing: [{miss_str}]"
            
            init_axis(ax3, hint_title, show_grid=True) # 开启网格

            # 画地面 (右 - 基于 Hint 自己的范围)
            plot_xzPlane(
                ax3,
                hint_MINS[0] - hint_trajec[index, 0], hint_MAXS[0] - hint_trajec[index, 0],
                0,
                hint_MINS[2] - hint_trajec[index, 1], hint_MAXS[2] - hint_trajec[index, 1],
            )
            
            # 画骨骼 (右 - 蓝色系) - [关键修改] 跳过无效连线，避免画出连接到原点的线
            # 使用蓝色区分，方便对比
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors_blue)):
                # chain 包含一系列相连的节点索引
                # 我们需要确保线段两端的节点都是有效的
                valid_chain_segments = []
                current_segment = []
                
                for j in chain:
                    if is_valid_joint(hint_local[index, j]):
                        current_segment.append(j)
                    else:
                        if len(current_segment) > 1:
                            valid_chain_segments.append(current_segment)
                        current_segment = []
                if len(current_segment) > 1:
                    valid_chain_segments.append(current_segment)
                
                linewidth = 4.0 if i < 5 else 2.0
                
                # 分段绘制有效连线
                for seg in valid_chain_segments:
                    ax3.plot3D(
                        hint_local[index, seg, 0],
                        hint_local[index, seg, 1],
                        hint_local[index, seg, 2],
                        linewidth=linewidth,
                        color=color,
                    )
            
            
            # 关节编号 (右) - [关键修改] 只标注有效节点
            for j in range(hint_local.shape[1]):
                if is_valid_joint(hint_local[index, j]):
                    ax3.text(
                        hint_local[index, j, 0], 
                        hint_local[index, j, 1], 
                        hint_local[index, j, 2], 
                    str(j), fontsize=6, color='black'
                )

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    
    update(int(frame_number / 2)) # 更新到中间帧抽样，方便观察整体效果
    plt.savefig(save_path.replace(".mp4", "_debug.png"), dpi=1000)
    
    plt.close()