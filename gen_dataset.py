"""
生成 tol=1.0 标准下的 200 个合格样本，参数展平存为 CSV
"""
import numpy as np
import pandas as pd
import time
from aircraft_gene import Aircraft

# ======================== 配置 ========================
BASELINE_CSV = "baseline_6tip.csv"
PERTURB_RATE = 0.20
N_SAMPLES = 500
OUTPUT_CSV = "dataset_tol1.0_500.csv"
MAX_ATTEMPTS = 100000

# ======================== 基准 ========================
bwb_base = Aircraft()
bwb_base.read_from_csv(BASELINE_CSV)
bwb_base.gene_simple_mesh(81, 81, aoa=0.0)
base_score = bwb_base.surface_smoothness(19)
base_cabin = bwb_base.search_cabin()
print(f"baseline surface_smoothness: {base_score:.4f}")
print(f"baseline search_cabin:        {base_cabin:.6f}")
print(f"阈值 (tol=1.0):              ≤ {base_score:.2f}")

# ======================== 筛选 ========================
rows = []
attempt = 0
t0 = time.time()

while len(rows) < N_SAMPLES:
    attempt += 1

    new_para = bwb_base.origin_para.copy()
    new_para[:, 1:19] *= (1.0 + np.random.uniform(-PERTURB_RATE, PERTURB_RATE,
                                                   size=new_para[:, 1:19].shape))
    new_para[1:, -5] += np.random.uniform(-0.5, 0.5, size=len(new_para) - 1)
    new_para[1:, -3] += np.random.uniform(-0.5, 0.5, size=len(new_para) - 1)

    bwb = Aircraft(new_para)

    if bwb.check_le_te() > 0:
        continue
    if np.max(new_para[:, -3]) > 2.1 or np.min(new_para[2:, -3]) < 0.5:
        continue
    if bwb.check_z_offset() > 0:
        continue

    bwb.gene_simple_mesh(81, 81, aoa=0.0)
    if bwb.search_cabin() >= 0.01:
        continue

    score = bwb.surface_smoothness(19)
    if score <= base_score * 1.0:
        # panel 网格生成检测（不保存）
        try:
            bwb.gene_panel_mesh(aoa=0.0)
        except Exception:
            continue
        rows.append(np.append(new_para.flatten(), score))
        n = len(rows)
        if n % 20 == 0:
            print(f"[{n}/{N_SAMPLES}] attempt={attempt}, score={score:.1f}")

# ======================== 保存 ========================
n_params = bwb_base.origin_para.size  # 8行 × 24列 = 192
cols = [f'p{i}' for i in range(n_params)] + ['smoothness_score']
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUTPUT_CSV, index=False)

dt = time.time() - t0
print(f"\n完成: {len(rows)} 样本, {attempt} 次尝试, {dt:.1f}s")
print(f"保存至: {OUTPUT_CSV}")
print(f"shape: {df.shape}")
