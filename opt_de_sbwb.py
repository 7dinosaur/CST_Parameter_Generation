"""
差分进化优化 — 匹配等效截面积分布
==============================================
基准:  mesh_para/new_sbwb.csv
目标:  target/Area_T.dat 中的 Area_T 曲线
约束:  客舱不侵入 / 升力 >= 1,190,000 N / 光顺值 <= 基准的 1.05 倍

工作流:
  apply_design_vars(x) → Aircraft → gene_simple_mesh (约束检查)
  → write_mesh("panel") → cal_Lift (FABOOM) → 读取 Total_equivalent_area.dat
  → 计算与 Area_T 的积分面积差

设计变量 (20 维, 5 个展向控制站 x 4 组参数):
  x[0:5]   = z_offset 增量 (m),       控制站 y = [0, 2, 6, 10, 14]
  x[5:10]  = dy 缩放因子,             控制站同上
  x[10:15] = 上表面 CST 系数缩放因子,  控制站同上
  x[15:20] = 下表面 CST 系数缩放因子,  控制站同上
  各控制站间的参数通过二次插值得到全部 11 个截面的扰动值
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import os
import time
import sys

from aircraft_gene_2 import Aircraft
from cal_Lift import cal_Lift

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_CSV = os.path.join(BASE_DIR, "mesh_para", "new_sbwb.csv")
TARGET_FILE = os.path.join(BASE_DIR, "target", "Area_T.dat")
FABOOM_DIR = "FABOOM_test"

# ============================================================
# 预计算：基准参数、基准光顺值、目标曲线
# ============================================================

# 读取基准参数
baseline_para = pd.read_csv(BASELINE_CSV).to_numpy()
N_SECTIONS = baseline_para.shape[0]
CST_ORDER = int(0.5 * (baseline_para.shape[1] - 8))
N_COEFFS = CST_ORDER + 1  # 单面 CST 系数个数

print(f"基准文件: {BASELINE_CSV}")
print(f"截面数: {N_SECTIONS}, CST 阶数: {CST_ORDER}, 总列数: {baseline_para.shape[1]}")

# 基准光顺值
base_air = Aircraft()
base_air.read_from_csv(BASELINE_CSV)
base_air.gene_simple_mesh(41, 41)
BASE_SMOOTH = np.array(base_air.if_smooth())
print(f"基准光顺值: upper = {BASE_SMOOTH[0]:.4f}, lower = {BASE_SMOOTH[1]:.4f}")

# 目标曲线
target_raw = np.loadtxt(TARGET_FILE)
TARGET_X = target_raw[:, 0]
TARGET_Y = target_raw[:, 1]
print(f"目标曲线: {len(TARGET_X)} 点, x ∈ [{TARGET_X.min():.1f}, {TARGET_X.max():.1f}], "
      f"max Ae = {TARGET_Y.max():.2f}")

# 控制站: 5 个展向位置, 覆盖翼根到翼尖
CTRL_IDX = np.array([0, 2, 5, 8, 10])  # 对应 y = 0, 2, 6, 10, 14
CTRL_Y = baseline_para[CTRL_IDX, 0]
ALL_Y = baseline_para[:, 0]
N_CTRL = len(CTRL_IDX)
N_VARS = N_CTRL * 4  # 20

print(f"控制站 y = {CTRL_Y}")
print(f"设计变量数: {N_VARS}")


# ============================================================
# 等效截面积差值计算
# ============================================================

def calc_area_diff(folder_path):
    """读取 Total_equivalent_area.dat 并与目标 Area_T 比较, 返回积分面积差"""
    area_file = os.path.join(folder_path, "area", "Total_equivalent_area.dat")
    if not os.path.exists(area_file):
        return 1e10

    data = np.loadtxt(area_file)
    x2, y2 = data[:, 0], data[:, 1]

    x_min = max(TARGET_X.min(), x2.min())
    x_max = min(TARGET_X.max(), x2.max())
    x_new = np.linspace(x_min, x_max, 1000)

    f1 = interp1d(TARGET_X, TARGET_Y, kind='linear', fill_value='extrapolate')
    f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')

    dy = np.abs(f1(x_new) - f2(x_new))
    return np.trapezoid(dy, x_new)


# ============================================================
# 设计变量 → 参数矩阵
# ============================================================

def apply_design_vars(x, base_para):
    """
    将 20 维设计变量应用到基准参数上.

    x 结构:
      x[0:5]   - z_offset 增量 (m)
      x[5:10]  - dy 缩放因子
      x[10:15] - 上表面 CST 缩放因子
      x[15:20] - 下表面 CST 缩放因子

    各控制站之间通过二次插值得到全部截面的扰动值.
    """
    new_para = base_para.copy()

    # --- z_offset 增量 (col -3) ---
    f_z = interp1d(CTRL_Y, x[0:5], kind='quadratic', fill_value='extrapolate')
    new_para[:, -3] += f_z(ALL_Y)

    # --- dy 缩放 (col -2, -1) ---
    f_dy = interp1d(CTRL_Y, x[5:10], kind='quadratic', fill_value='extrapolate')
    scales_dy = f_dy(ALL_Y)
    new_para[:, -2] *= scales_dy
    new_para[:, -1] *= scales_dy

    # --- 上表面 CST 缩放 (col 1 : 1+N_COEFFS) ---
    f_cst_u = interp1d(CTRL_Y, x[10:15], kind='quadratic', fill_value='extrapolate')
    for i in range(N_COEFFS):
        new_para[:, 1 + i] *= f_cst_u(ALL_Y)

    # --- 下表面 CST 缩放 (col 1+N_COEFFS : 1+2*N_COEFFS) ---
    f_cst_l = interp1d(CTRL_Y, x[15:20], kind='quadratic', fill_value='extrapolate')
    for i in range(N_COEFFS):
        new_para[:, 1 + N_COEFFS + i] *= f_cst_l(ALL_Y)

    return new_para


# ============================================================
# 约束惩罚常量
# ============================================================
PENALTY_BIG = 1e8       # 致命错误 (FABOOM 崩溃 / 几何异常)
PENALTY_SMOOTH = 10.0   # 光顺超标: 每单位超出 * 此系数
PENALTY_CABIN = 5000.0  # 客舱侵入
LIFT_TARGET = 1190000.0 # 最小升力 (N), ≈ 122,500 kg * 9.81 * 0.99


# ============================================================
# 目标函数
# ============================================================

# 全局追踪变量
_best_f = np.inf
_best_x = None
_n_evals = 0
_start_time = None


def objective(x):
    """DE 目标函数: 返回积分面积差 (越小越好), 约束违规时返回大罚值"""
    global _best_f, _best_x, _n_evals, _start_time
    if _start_time is None:
        _start_time = time.time()
    _n_evals += 1

    # --- 1. 应用设计变量, 生成新参数 ---
    try:
        new_para = apply_design_vars(x, baseline_para)
    except Exception:
        return PENALTY_BIG

    # --- 2. 构建 Aircraft, 生成 simple_mesh 用于约束检查 ---
    try:
        air = Aircraft(new_para)
        air.gene_simple_mesh(41, 41)
    except Exception:
        return PENALTY_BIG

    # --- 3. 光顺约束: 上下表面均不能超过基准的 1.05 倍 ---
    s = np.array(air.if_smooth())
    smooth_violation = np.maximum(s - BASE_SMOOTH * 1.05, 0.0)
    if smooth_violation.sum() > 0:
        return PENALTY_BIG + smooth_violation.sum() * PENALTY_SMOOTH

    # --- 4. 客舱约束: 网格点侵入客舱体积的比例 < 1% ---
    if not air.search_cabin():
        return PENALTY_BIG + PENALTY_CABIN

    # --- 5. 写面元网格 ---
    try:
        geo_path = os.path.join(FABOOM_DIR, "indata", "geo.x")
        air.write_mesh("panel", geo_path, 0.0)
    except Exception:
        return PENALTY_BIG

    # --- 6. 运行 FABOOM ---
    try:
        Lift = cal_Lift(FABOOM_DIR)
    except Exception:
        return PENALTY_BIG

    if Lift is False:
        return PENALTY_BIG

    # --- 7. 升力约束 ---
    if Lift < LIFT_TARGET:
        return PENALTY_BIG + (LIFT_TARGET - Lift) * 1e-3

    # --- 8. 目标: 等效截面积积分差 ---
    obj = calc_area_diff(FABOOM_DIR)

    # --- 保存最优解 ---
    if obj < _best_f:
        _best_f = obj
        _best_x = x.copy()
        best_para = apply_design_vars(x, baseline_para)
        pd.DataFrame(best_para).to_csv(
            os.path.join(BASE_DIR, "best_de_para.csv"), index=False)
        np.save(os.path.join(BASE_DIR, "best_de_x.npy"), x)
        elapsed = time.time() - _start_time
        print(f"\n{'=' * 55}")
        print(f" [Eval {_n_evals}] 新最优解 | 耗时 {elapsed:.0f}s")
        print(f" 目标值 (面积差) = {obj:.6e}")
        print(f" 光顺: upper={s[0]:.4f} (基准 {BASE_SMOOTH[0]:.4f}), "
              f"lower={s[1]:.4f} (基准 {BASE_SMOOTH[1]:.4f})")
        print(f" 升力 = {Lift:.0f} N")
        print(f"{'=' * 55}")

    return obj


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 边界设置
    bounds = [
        *[(-0.5, 0.5)] * N_CTRL,    # z_offset 增量: ±0.5 m
        *[(0.5, 1.5)] * N_CTRL,     # dy 缩放: 0.5 ~ 1.5
        *[(0.8, 1.2)] * N_CTRL,     # 上表面 CST 缩放: 0.8 ~ 1.2
        *[(0.8, 1.2)] * N_CTRL,     # 下表面 CST 缩放: 0.8 ~ 1.2
    ]

    print(f"\n边界设置:")
    print(f"  z_offset : ±0.5 m")
    print(f"  dy       : 0.5 ~ 1.5 × baseline")
    print(f"  CST 系数 : 0.8 ~ 1.2 × baseline")
    print(f"\n开始差分进化优化... (workers=1 因 FABOOM 不可并行)")
    print(f"popsize=15, maxiter=200, 预计最多 {15 * 200} 次评估\n")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=200,
        tol=1e-4,
        workers=1,               # 必须串行, FABOOM 是外部进程
        disp=True,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=False,
    )

    print("\n" + "=" * 60)
    print("优化完成!")
    print(f"DE 返回最优值: {result.fun:.6e}")
    print(f"全局最优值:    {_best_f:.6e}")
    print(f"总评估次数:    {_n_evals}")
    print(f"总耗时:        {time.time() - _start_time:.0f}s")

    # 保存最终结果
    final_para = apply_design_vars(_best_x, baseline_para)
    final_path = os.path.join(BASE_DIR, "opt_de_final.csv")
    pd.DataFrame(final_para).to_csv(final_path, index=False)
    print(f"最终参数已保存至: {final_path}")

    # 输出最优参数变化摘要
    print(f"\n参数变化摘要 (最优 vs 基准):")
    print(f"{'y':>6s}  {'dz_off':>8s}  {'dy_scale':>9s}  "
          f"{'CST_u_scale':>11s}  {'CST_l_scale':>11s}")
    for i in range(N_SECTIONS):
        dz = final_para[i, -3] - baseline_para[i, -3]
        dy_s = final_para[i, -2] / baseline_para[i, -2] if baseline_para[i, -2] != 0 else 0
        cst_u_s = np.mean(final_para[i, 1:1+N_COEFFS] / np.abs(baseline_para[i, 1:1+N_COEFFS]))
        cst_l_s = np.mean(final_para[i, 1+N_COEFFS:1+2*N_COEFFS] / np.abs(baseline_para[i, 1+N_COEFFS:1+2*N_COEFFS]))
        print(f"{baseline_para[i, 0]:6.1f}  {dz:8.4f}  {dy_s:9.4f}  "
              f"{cst_u_s:11.4f}  {cst_l_s:11.4f}")
