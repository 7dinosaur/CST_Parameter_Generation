import os
import numpy as np
import pandas as pd

from aircraft_gene import Aircraft
from cal_Lift import cal_Lift
    
def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()
    total_cols = new_para.shape[1]
    keep_cols = [0, total_cols-5, total_cols-4]
    mask = np.ones(total_cols, dtype=bool)
    mask[keep_cols] = False  # 不动的列设为 False

    params_to_perturb = new_para[:, mask]
    perturb_factor = 1 + perturbation * (2 * np.random.rand(*params_to_perturb.shape) - 1)
    new_para[:, mask] = params_to_perturb * perturb_factor

    return new_para

def main():
    # ===================== 【你只需要改这3个配置】 =====================
    para_csv = "increase_cabin.csv"
    output_csv = "qualified_solutions.csv"  # 合格结果永久保存
    LIFT_MIN_THRESHOLD = 1170000.0  # 升力下限（必须修改为你的真实值）
    perturb_rate = 0.03           # 扰动幅度

    # ===================== 读取基础参数 =====================
    base_para = pd.read_csv(para_csv).to_numpy()
    param_count = len(base_para.flatten())  # 自动计算参数数量

    # ===================== 初始化输出文件（只写一次表头） =====================
    if not os.path.exists(output_csv):
        # 构造列名
        columns = ["iteration", "Lift", "passenger"] + [f"param_{i}" for i in range(param_count)]
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False, encoding="utf-8-sig")

    iteration = 0
    print("===== 无限循环启动！按 Ctrl + C 可安全停止，数据不会丢失 =====")

    # ===================== 【无限循环核心】 =====================
    while True:
        iteration += 1
        print(f"\n----- 第 {iteration} 次生成 -----")

        try:
            # 1. 扰动参数
            new_para = perturb_para(base_para, perturb_rate)

            # 2. 生成模型 & 计算
            new_air = Aircraft(new_para)
            new_air.write_mesh("panel", r"FABOOM_test\indata\geo.x")
            passenger = new_air.cal_volume()
            Lift = cal_Lift()

            # 3. 升力判断
            if not Lift or Lift < LIFT_MIN_THRESHOLD:
                print(f"❌ 不合格 | 升力: {Lift:.2f} (下限: {LIFT_MIN_THRESHOLD})")
                continue

            # 4. ✅ 合格：立刻保存到文件（实时写入，中断不丢）
            print(f"✅ 合格 | 升力: {Lift:.2f} | 载客量: {passenger:.2f}")
            
            # 拼接一行数据
            row = [iteration, Lift, passenger, *new_para.flatten()]
            
            # 追加写入（关键：不会丢失数据）
            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")

        # ===================== 手动停止：优雅退出 =====================
        except KeyboardInterrupt:
            print("\n🛑 手动中断程序！所有合格数据已保存，无丢失！")
            break

        # ===================== 其他错误：跳过本次，不崩溃 =====================
        except Exception as e:
            print(f"⚠️  计算出错，已跳过：{str(e)}")
            continue

if __name__ == "__main__":
    main()