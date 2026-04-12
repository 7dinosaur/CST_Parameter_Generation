import os
import numpy as np
import pandas as pd
import time

from aircraft_gene import Aircraft
from cal_Lift import cal_Lift
    
def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()

    cst_cols = slice(1, 19)
    params_to_perturb = new_para[:, cst_cols]
    perturb_factor = perturbation * (2 * np.random.rand(*params_to_perturb.shape) - 1)
    ## 对CST参数进行扰动
    new_para[:, cst_cols] = params_to_perturb + perturb_factor
    ## 对剖面Z方向偏移进行扰动
    new_para[1:, -3] = new_para[1:, -3] + (np.random.rand(*new_para[1:, -3].shape) - 0.5) * 0.1
    ## 对剖面的后缘z方向偏移进行扰动
    # new_para[:, -2:] = new_para[:, -2:] + (np.random.rand(*new_para[:, -2:].shape) - 0.5) * 0.2
    ## 对剖面x方向偏移进行扰动
    # new_para[1:, -5:-3] = new_para[1:, -5:-3] + (np.random.rand(*new_para[1:, -5:-3].shape) - 0.5) * 0.5

    return new_para

def main():
    para_csv = r"mesh_para\\6.64_simple.csv"
    output_csv = r"database\samples_based_165_664.csv"
    LIFT_MIN_THRESHOLD = 1200000.0  # 升力下限
    LIFT_MAX_THRESHOLD = 1500000.0  # 升力上限
    passenger_min = 160
    perturb_rate = 0.02           # 扰动幅度

    base_para = pd.read_csv(para_csv).to_numpy()
    param_count = len(base_para.flatten())  # 自动计算参数数量

    if not os.path.exists(output_csv):
        # 构造列名
        columns = ["iteration", "passenger"] + [f"param_{i}" for i in range(param_count)]
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False, encoding="utf-8-sig")

    iteration = 0

    base_air = Aircraft(base_para)
    base_laplace = base_air.Laplace()
    btu, btl = base_air.if_smooth()
    pass_count = 0

    t0 = time.time()
    while True:
        iteration += 1
        
        # print(f"\n----- 第 {iteration} 次生成 -----")

        try:
            # 1. 扰动参数
            new_para = perturb_para(base_para, perturb_rate)

            # 2. 生成模型 & 计算
            new_air = Aircraft(new_para)
            l1 = np.array(new_air.Laplace())
            tu, tl = new_air.if_smooth()
            if (l1 > np.array(base_laplace) * 1.0).any() or tu > 6 or tl > 6:  # 几何光顺性判断（阈值可调整）
            # if (l1 > np.array(base_laplace) * 1.2).any():  # 几何光顺性判断（阈值可调整）
                # print(f"❌ 几何光顺不合格")
                continue
            # print("正在计算载客量")
            passenger = new_air.cal_volume()
            # 载客量判断
            if passenger < passenger_min:
                print(f"❌ 不合格 | 载客量 {passenger}")
                continue
            t1 = time.time()
            print(f"✅ 第{pass_count + 1}个合格样本 | 载客量: {passenger:.2f} | 耗时：{t1-t0}")
            pass_count += 1
            # new_air.write_mesh("panel", r"FABOOM_test\indata\geo.x", aoa=4.0)
            # Lift = cal_Lift()

            # 升力判断
            # if not Lift or Lift < LIFT_MIN_THRESHOLD or Lift > LIFT_MAX_THRESHOLD:
            #     print(f"❌ 不合格 | 升力: {Lift:.2f}")
                # continue

            # 4. ✅ 合格：立刻保存到文件（实时写入，中断不丢）
            # print(f"✅ 合格 | 升力: {Lift:.2f} | 载客量: {passenger:.2f}")
            
            # 拼接一行数据
            row = [iteration, passenger, *new_para.flatten()]
            
            # 追加写入（关键：不会丢失数据）
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")

        except KeyboardInterrupt:
            print("\n🛑 手动中断程序！所有合格数据已保存，无丢失！")
            break

        # except Exception as e:
        #     print(f"⚠️  计算出错，已跳过：{str(e)}")
        #     continue

if __name__ == "__main__":
    main()