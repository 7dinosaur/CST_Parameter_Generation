import os
import numpy as np
import pandas as pd

from aircraft_gene import Aircraft
from cal_Lift import cal_Lift
    
def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()

    cst_cols = slice(1, 19)
    params_to_perturb = new_para[:, cst_cols]
    perturb_factor = perturbation * (2 * np.random.rand(*params_to_perturb.shape) - 1)
    new_para[:, cst_cols] = params_to_perturb + perturb_factor
    new_para[:, -3] = new_para[:, -3] + (np.random.rand(*new_para[:, -3].shape) - 0.5) * 0.2

    return new_para

def main():
    para_csv = "smooth_test.csv"
    output_csv = "qualified_solutions_1.csv"
    LIFT_MIN_THRESHOLD = 1100000.0  # 升力下限
    LIFT_MAX_THRESHOLD = 1800000.0  # 升力上限
    passenger_min = 120
    perturb_rate = 0.02           # 扰动幅度

    base_para = pd.read_csv(para_csv).to_numpy()
    param_count = len(base_para.flatten())  # 自动计算参数数量

    if not os.path.exists(output_csv):
        # 构造列名
        columns = ["iteration", "Lift", "passenger"] + [f"param_{i}" for i in range(param_count)]
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False, encoding="utf-8-sig")

    iteration = 0

    while True:
        iteration += 1
        print(f"\n----- 第 {iteration} 次生成 -----")

        try:
            # 1. 扰动参数
            new_para = perturb_para(base_para, perturb_rate)

            # 2. 生成模型 & 计算
            new_air = Aircraft(new_para)
            if not new_air.if_smooth():
                # print(f"❌ 几何光顺不合格")
                continue
            new_air.write_mesh("panel", r"FABOOM_test\indata\geo.x", aoa=3.6)
            print("正在计算载客量")
            passenger = new_air.cal_volume()
            # 载客量判断
            if passenger < passenger_min:
                print(f"❌ 不合格 | 载客量 {passenger}")
                continue
            Lift = cal_Lift()

            # 升力判断
            if not Lift or Lift < LIFT_MIN_THRESHOLD or Lift > LIFT_MAX_THRESHOLD:
                print(f"❌ 不合格 | 升力: {Lift:.2f}")
                continue

            # 4. ✅ 合格：立刻保存到文件（实时写入，中断不丢）
            print(f"✅ 合格 | 升力: {Lift:.2f} | 载客量: {passenger:.2f}")
            
            # 拼接一行数据
            row = [iteration, Lift, passenger, *new_para.flatten()]
            
            # 追加写入（关键：不会丢失数据）
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")

        except KeyboardInterrupt:
            print("\n🛑 手动中断程序！所有合格数据已保存，无丢失！")
            break

        except Exception as e:
            print(f"⚠️  计算出错，已跳过：{str(e)}")
            continue

if __name__ == "__main__":
    main()