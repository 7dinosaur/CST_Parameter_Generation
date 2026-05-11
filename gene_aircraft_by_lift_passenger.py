import numpy as np
import pandas as pd
import time

from aircraft_gene import Aircraft
    
def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()

    new_para[:, 1:-5] *= (1 + np.random.uniform(-perturbation, perturbation, size=new_para[:, 1:-5].shape))
    # new_para[:, -1] = new_para[:, -2]

    return new_para

def MCMC_samples(have_data=False):
    para_csv = r"mesh_para\\tmp_bwb.csv"
    output_csv = r"database\smooth_history.csv"
    passenger_min = 160
    perturb_rate = 0.05           # 扰动幅度

    ## 读取基准外形
    base_para = pd.read_csv(para_csv).to_numpy()
    print(base_para.shape)

    if not have_data:
        ## 将基准外形写入可行解集
        row = [*base_para.flatten()]
        pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")

    iteration = 0

    t0 = time.time()
    pass_count = 0
    while True:
        iteration += 1

        try:
            # 1. 扰动参数
            base_para = pd.read_csv(output_csv, header=None).to_numpy()[-1].reshape([*base_para.shape])  # 每次迭代都从文件读取最新的参数，确保扰动基于最新数据
            base_u = Aircraft(base_para).if_smooth()
            while True:
                new_para = perturb_para(base_para, perturb_rate)

                # 2. 生成模型 & 计算
                new_air = Aircraft(new_para)
                new_u = new_air.if_smooth()

                if (new_u > base_u*0.99).any():  # 几何光顺性判断（阈值可调整）
                    print(f"❌ 几何光顺不合格")
                    continue
                # print("正在计算载客量")
                # passenger = new_air.cal_volume()
                # 载客量判断
                if not new_air.search_cabin():
                    print(f"❌ 不合格 | 载客量不足")
                    continue
                t1 = time.time()
                print("本次用时：" + str(t1-t0) + "秒")
                t0 = t1
                print(f"✅ 第{pass_count + 1}个合格样本 |  | 耗时：{t1-t0}")
                pass_count += 1
            
                # 拼接一行数据
                row = [*new_para.flatten()]
                
                # 追加写入（关键：不会丢失数据）
                pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")

        except KeyboardInterrupt:
            print("\n🛑 手动中断程序！所有合格数据已保存，无丢失！")
            break

def test_sample():
    database = pd.read_csv(r"database\MCMC.csv", header=None).to_numpy()
    print(database.shape)
    ### 随机取一个样本生成test.x
    sample_para = database[np.random.randint(0, database.shape[0])].reshape([10, 18])
    test_air = Aircraft(sample_para)
    test_air.write_mesh("simple", "test.x", 0)


if __name__ == "__main__":
    MCMC_samples(have_data=False)
    # test_sample()