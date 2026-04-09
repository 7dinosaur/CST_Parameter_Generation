import numpy as np
import pandas as pd
import os

from aircraft_gene import Aircraft
from cal_Lift import cal_Lift

if __name__ == "__main__":
    para = pd.read_csv(r"database\samples_based_bwb3.csv").to_numpy()[:, 2:]

    # 结果文件路径
    result_file = r"calculation_results.csv"

    # 先初始化一个空 DataFrame 并写入表头（如果文件不存在）
    try:
        # 尝试读取，文件不存在会报错
        pd.read_csv(result_file)
    except FileNotFoundError:
        # 文件不存在，创建带表头的空文件
        df_empty = pd.DataFrame(columns=['index', 'tu', 'tl', 'passenger', 'Lift'])
        df_empty.to_csv(result_file, index=False, encoding='utf-8-sig')

    for idx, pa in enumerate(para):
        air_para = Aircraft(pa.reshape([-1, 24]))
        air_para.write_mesh("panel", r"FABOOM_test\indata\geo.x", 3.8)
        tu, tl = air_para.if_smooth()
        passenger = air_para.cal_volume()
        Lift = cal_Lift()

        # 构造一行结果
        row = {
            'index': idx,
            'tu': tu,
            'tl': tl,
            'passenger': passenger,
            'Lift': Lift
        }

        # 追加写入，不覆盖
        pd.DataFrame([row]).to_csv(
            result_file,
            mode='a',
            header=False,
            index=False,
            encoding='utf-8-sig'
        )

        # 打印看一下当前结果
        print(f"索引 {idx} 已写入 | tu={tu}, tl={tl}, 乘客数={passenger}, 升力={Lift}")
        
    