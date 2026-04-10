from aircraft_gene import Aircraft
from cal_Lift import cal_Lift

import numpy as np
import pandas as pd
from numpy import random

def D_area(folder_path):
    # 读取dat文件
    def read_curve(file_path):
        data = np.loadtxt(file_path,skiprows=1)
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    # 读取两条曲线的数据
    x1, y1 = read_curve(r'target\Area_T.dat')
    x2, y2 = read_curve(f'{folder_path}\\area\\Total_equivalent_area.dat')

    # 生成统一的新x轴序列（覆盖两条曲线的范围）
    x_min = max(x1.min(), x2.min())
    x_max = min(x1.max(), x2.max())
    x_new = np.linspace(x_min, x_max, 1000)  # 1000个点，按需调整

    # 创建插值函数（线性插值，可改为'cubic'）
    f1 = interp1d(x1, y1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(x2, y2, kind='linear', fill_value='extrapolate')

    # 插值到新x轴
    y1_new = f1(x_new)
    y2_new = f2(x_new)

    # 计算差值
    delta_y = np.abs(y1_new - y2_new)
    area_trapz = np.trapezoid(delta_y, x_new)

    return area_trapz

if __name__ == "__main__":
    para = pd.read_csv(r"database\samples_based_bwb3.csv").to_numpy()[:, 2:]

    result = para.shape[0] * random.rand(2)
    n1 = int(result[0]); n2 = int(result[1]); ratio = result[0] - n1
    print(result)
    print(n1, n2, ratio)
    para1 = para[n1].reshape([-1, 24]); para2 = para[n2].reshape([-1, 24])
    new_para = para1*ratio + para2*(1 - ratio)
    para_list = [para1, para2, new_para]
    air_list = [Aircraft(n) for n in para_list]
    print([a.if_smooth() for a in air_list])
    print([a.Laplace() for a in air_list])
    print([a.cal_volume() for a in air_list])
    air_list[-1].write_mesh("panel", "check.x")