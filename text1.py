# import os
# from pathlib import Path

# root = Path("dataset/HumanML3D")
# print("data_root exists:", root.exists(), root.resolve())

# # 检查关键文件
# for p in ["Mean.npy", "Std.npy"]:
#     print(p, (root/p).exists())

# # 检查特征目录
# cands = ["new_joint_vecs", "new_joint_vecs/"]
# for c in cands:
#     d = root / c
#     if d.exists():
#         files = list(d.glob("*.npy"))
#         print("found", len(files), "feature files in", d)
#         if files:
#             print("example:", files[0].name)

# # 检查 split 文件常见位置
# for p in ["train.txt","val.txt","test.txt","splits/train.txt","splits/test.txt","splits/val.txt"]:
#     fp = root/p
#     if fp.exists():
#         print("split file:", fp, "lines:", sum(1 for _ in open(fp, "r", encoding="utf-8", errors="ignore")))


from pathlib import Path

root = Path("dataset/HumanML3D").resolve()
ids = (root/"train.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
first = ids[0].strip()
print("first id:", first)

candidates = [
    root/"new_joint_vecs"/f"{first}.npy",
    root/"new_joint_vecs"/f"M{first}.npy",
    root/"new_joints"/f"{first}.npy",
    root/"new_joints"/f"M{first}.npy",
    root/"motions"/f"{first}.npy",
    root/"motions"/f"M{first}.npy",
]

for p in candidates:
    print(str(p), "exists=", p.exists())