"""
随机扰动 CST 系数，按 surface_smoothness 三个容忍度筛选外形
tol=1.0 / 1.1 / 1.2，各收集 10 个合格外形，输出 simple 网格 .x 文件
"""
import numpy as np
import os
import time
from aircraft_gene import Aircraft

# ======================== 配置 ========================
BASELINE_CSV = "baseline_6tip.csv"
PERTURB_RATE = 0.20          # CST 系数扰动幅度
N_PER_TOL = 10               # 每个容忍度收集数量
TOLERANCES = [1.0, 1.1, 1.2]
OUTPUT_ROOT = "smooth_check"
MAX_ATTEMPTS = 20000

# ======================== 基准 ========================
bwb_base = Aircraft()
bwb_base.read_from_csv(BASELINE_CSV)
bwb_base.gene_simple_mesh(81, 81, aoa=0.0)
base_score = bwb_base.surface_smoothness(19)
base_cabin = bwb_base.search_cabin()
print(f"基准 surface_smoothness(19): {base_score:.4f}")
print(f"基准 search_cabin:           {base_cabin:.6f}")

# 创建输出目录
for tol in TOLERANCES:
    os.makedirs(f"{OUTPUT_ROOT}/tol_{tol}", exist_ok=True)

# ======================== 筛选循环 ========================
counts = {t: 0 for t in TOLERANCES}
attempt = 0
t0 = time.time()

while attempt < MAX_ATTEMPTS:
    attempt += 1

    # 1. 扰动 CST 系数 + 前缘 + z偏移（锁定后缘和y站位）
    new_para = bwb_base.origin_para.copy()
    noise = 1.0 + np.random.uniform(-PERTURB_RATE, PERTURB_RATE,
                                     size=new_para[:, 1:19].shape)
    new_para[:, 1:19] *= noise
    new_para[1:, -5] += np.random.uniform(-0.2, 0.2, size=len(new_para) - 1)   # le
    new_para[1:, -3] += np.random.uniform(-0.1, 0.1, size=len(new_para) - 1)   # z_offset

    bwb_new = Aircraft(new_para)

    # 2. 前后缘硬约束
    if bwb_new.check_le_te() > 0:
        continue

    # 3. z_offset 约束：最大值 ≤ 2.1m + 单调递增（防鼓包）
    if np.max(new_para[:, -3]) > 2.1:
        continue
    if bwb_new.check_z_offset() > 0:
        continue

    # 4. 客舱约束
    bwb_new.gene_simple_mesh(81, 81, aoa=0.0)
    if bwb_new.search_cabin() >= 0.01:
        continue

    # 5. 光顺度 + 分类
    score = bwb_new.surface_smoothness(19)

    for tol in TOLERANCES:
        if counts[tol] >= N_PER_TOL:
            continue
        if score <= base_score * tol:
            idx = counts[tol]
            path = f"{OUTPUT_ROOT}/tol_{tol}/{idx}.x"
            bwb_new.write_mesh("simple", path, 0)
            counts[tol] += 1
            print(f"[tol={tol}] {idx+1}/{N_PER_TOL}  score={score:.3f}  "
                  f"(base*{tol}={base_score*tol:.3f})  attempt={attempt}")
            break  # 每个样本只归入最严格的那一档

    # 全部收集完毕
    if all(v >= N_PER_TOL for v in counts.values()):
        break

# ======================== 结果 ========================
dt = time.time() - t0
print(f"\n{'='*50}")
print(f"完成。总尝试 {attempt} 次，耗时 {dt:.1f}s")
for tol in TOLERANCES:
    folder = f"{OUTPUT_ROOT}/tol_{tol}"
    files = os.listdir(folder) if os.path.isdir(folder) else []
    print(f"  tol={tol}: {len(files)} 个文件 → {folder}/")
