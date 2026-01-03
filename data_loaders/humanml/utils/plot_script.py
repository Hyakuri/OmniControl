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
