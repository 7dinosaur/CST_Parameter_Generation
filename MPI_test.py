from concurrent.futures import ProcessPoolExecutor
from cal_Lift import cal_Lift
import numpy as np
import os
import pandas as pd
from aircraft_gene import Aircraft

BASE_DIRS = [
    "MPI_FABOOM\\FABOOM_01",
    "MPI_FABOOM\\FABOOM_02",
    "MPI_FABOOM\\FABOOM_03",
    "MPI_FABOOM\\FABOOM_04"
]  # 4个并行计算目录
PARA_CSV = "opt1_99.4_144.csv"
OUTPUT_CSV = "new_samples_parallel.csv"  # 最终统一输出文件
LIFT_MIN_THRESHOLD = 1200000.0
LIFT_MAX_THRESHOLD = 1500000.0
PASSENGER_MIN = 120
PERTURB_RATE = 0.03
GEOMETRY_THRESHOLD = 1.05  # 几何光顺阈值
perturb_rate = 0.03
passenger_min = 120

base_para = pd.read_csv(PARA_CSV).to_numpy()
param_count = len(base_para.flatten())
base_air = Aircraft(base_para)
base_laplace = base_air.Laplace()

def perturb_para(base_para, perturbation=0.05):
    """参数扰动函数（与原代码完全一致）"""
    new_para = base_para.copy()
    cst_cols = slice(1, 19)
    params_to_perturb = new_para[:, cst_cols]
    perturb_factor = perturbation * (2 * np.random.rand(*params_to_perturb.shape) - 1)
    new_para[:, cst_cols] = params_to_perturb + perturb_factor
    new_para[:, -3] = new_para[:, -3] + (np.random.rand(*new_para[:, -3].shape) - 0.5) * 0.2
    new_para[:, -5:-3] = new_para[:, -5:-3] + (np.random.rand(*new_para[:, -5:-3].shape) - 0.5) * 0.2
    return new_para

# 直接并行运行，不需要子解释器
def run_task(interp_id):
    print(f"开始运行任务 {interp_id}")
    OUTPUT_CSV = f"new_samples_parallel_{interp_id}.csv"  # 每个进程独立输出文件
    if not os.path.exists(OUTPUT_CSV):
        columns = ["iteration", "Lift", "passenger"] + [f"param_{i}" for i in range(param_count)]
        pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    iteration = 0
    while True:
        iteration += 1
        print(f"\n----- 进程 {interp_id} 第 {iteration} 次生成 -----")
        new_para = perturb_para(base_para, perturb_rate)
        # 2. 生成模型 & 计算
        new_air = Aircraft(new_para)
        if new_air.Laplace() > base_laplace * 1.05:  # 几何光顺性判断（阈值可调整）
            # print(f"❌ 几何光顺不合格")
            continue
        print("正在计算载客量")
        passenger = new_air.cal_volume()
        # 载客量判断
        if passenger < passenger_min:
            print(f"❌ 不合格 | 载客量 {passenger}")
            continue

        new_air.write_mesh("panel", os.path.join(BASE_DIRS[interp_id - 1], "indata", "geo.x"), aoa=4.0)
        Lift = cal_Lift(base_path=BASE_DIRS[interp_id - 1])

        # 升力判断
        if not Lift or Lift < LIFT_MIN_THRESHOLD or Lift > LIFT_MAX_THRESHOLD:
            print(f"❌ 不合格 | 升力: {Lift:.2f}")
            continue

        # 4. ✅ 合格：立刻保存到文件（实时写入，中断不丢）
        print(f"✅ 合格 | 升力: {Lift:.2f} | 载客量: {passenger:.2f}")
        
        # 拼接一行数据
        row = [iteration, Lift, passenger, *new_para.flatten()]
        
        # 追加写入（关键：不会丢失数据）
        pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    task_list = [1, 2, 3, 4]  # 任务列表
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 提交任务
        futures = [executor.submit(run_task, tid) for tid in task_list]