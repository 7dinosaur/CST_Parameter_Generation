from concurrent.futures import ProcessPoolExecutor

from matplotlib.pylab import laplace
from cal_Lift import cal_Lift
import numpy as np
import os
import pandas as pd
from aircraft_gene import Aircraft
import time

# ===================== 配置 =====================
BASE_DIRS = [
    "MPI_FABOOM\\FABOOM_01",
    "MPI_FABOOM\\FABOOM_02",
    "MPI_FABOOM\\FABOOM_03",
    "MPI_FABOOM\\FABOOM_04"
]
PARA_CSV = "smooth_test.csv"
OUTPUT_CSV = "final_valid_samples.csv"
LIFT_MIN = 1200000.0
LIFT_MAX = 1500000.0
PASSENGER_MIN = 120
PERTURB_RATE = 0.03
GEOMETRY_FACTOR = 1.0

BATCH_SIZE = 4  # 每4个一组并行气动计算

# ===================== 工具函数 =====================
def task_wrapper(args):
    """
    顶层函数！！！
    多进程只能序列化顶层函数，不能序列化嵌套函数
    """
    idx, sample_list, dir_list = args
    para, _, _ = sample_list[idx]
    base_path = dir_list[idx]
    
    air = Aircraft(para)
    mesh_file = os.path.join(base_path, "indata", "geo.x")
    air.write_mesh("panel", mesh_file, aoa=4.0)
    
    return cal_Lift(base_path=base_path)

def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()
    cst_cols = slice(1, 19)
    params = new_para[:, cst_cols]
    perturb = perturbation * (2 * np.random.rand(*params.shape) - 1)
    new_para[:, cst_cols] = params + perturb
    new_para[:, -3] += (np.random.rand(*new_para[:, -3].shape) - 0.5) * 0.2
    new_para[:, -5:-3] += (np.random.rand(*new_para[:, -5:-3].shape) - 0.5) * 0.2
    return new_para

def generate_one_candidate(base_para, base_laplace):
    """生成 1 个【几何+载客量合格】的候选样本（极快）"""
    while True:
        new_para = perturb_para(base_para, PERTURB_RATE)
        air = Aircraft(new_para)
        laplace = air.Laplace()
        if laplace[0] > base_laplace[0] * GEOMETRY_FACTOR or laplace[1] > base_laplace[1] * GEOMETRY_FACTOR:
            continue
        passenger = air.cal_volume()
        if passenger >= PASSENGER_MIN:
            return new_para, passenger, laplace

def parallel_calc_lift(sample_list, dir_list):
    """并行气动计算（修复版）"""
    # 构造参数
    task_args = [
        (0, sample_list, dir_list),
        (1, sample_list, dir_list),
        (2, sample_list, dir_list),
        (3, sample_list, dir_list),
    ]
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        lift_list = list(executor.map(task_wrapper, task_args))
    return lift_list

# ===================== 主流程 =====================
if __name__ == "__main__":
    base_para = pd.read_csv(PARA_CSV).to_numpy()
    base_air = Aircraft(base_para)
    base_laplace = base_air.Laplace()
    param_count = len(base_para.flatten())

    # 初始化输出CSV
    if not os.path.exists(OUTPUT_CSV):
        cols = ["iteration", "Lift", "passenger"] + [f"param_{i}" for i in range(param_count)]
        pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total_iter = 0
    valid_count = 0

    print("🚀 开始生成 + 轻量筛选 + 并行气动计算")

    while True:
        batch = []
        # 1. 先生成 4 个轻量合格样本（超快）
        while len(batch) < BATCH_SIZE:
            try:
                para, pas = generate_one_candidate(base_para, base_laplace)
                batch.append((para, pas, total_iter))
                total_iter += 1
                print(f"[{total_iter}] 候选样本生成完成 | 载客={pas}")
            except Exception as e:
                print("生成出错，跳过")
                continue

        print("\n🔥 4个一组，开始并行气动计算...")
        t0 = time.time()
        lift_results = parallel_calc_lift(batch, BASE_DIRS)
        print(f"✅ 并行气动计算完成，耗时 {time.time()-t0:.2f}s\n")

        # 3. 最终筛选 + 保存
        for i in range(4):
            para, pas, it = batch[i]
            lift = lift_results[i]
            if lift and LIFT_MIN < lift < LIFT_MAX:
                valid_count += 1
                row = [valid_count, lift, pas, *para.flatten()]
                pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding="utf-8-sig")
                print(f"[{valid_count}] ✅ 最终合格 | 升力={lift:.2f}")
            else:
                print(f"❌ 升力不合格 | {lift}")