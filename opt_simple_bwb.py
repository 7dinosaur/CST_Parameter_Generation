from aircraft_gene import Aircraft
from cal_Lift import cal_Lift

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

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

def perturb_para(x, base_para):
    base_air = Aircraft(base_para)
    base_air.write_mesh("simple", "tmp.x")
    base_smooth = np.array(base_air.if_smooth())
    new_para = base_para.copy()
    x = x.reshape([2, 5])
    new_para[2:, -3] += x[0]
    new_para[2:, -2:] += (0.02*x[1]).reshape([5, 1])
    new_air = Aircraft(new_para)
    new_air.write_mesh("panel", r"FABOOM_test\indata\geo.x", 0.0)
    new_air.write_mesh("simple", "tmp.x")
    if (np.array(new_air.if_smooth()) > base_smooth*1.1).any():
        return 1000
    print(new_air.Laplace())
    try:
        Lift = cal_Lift()
    except:
        return 1000
    
    obj = D_area(r"FABOOM_test")
    return obj

best_solution = None
best_fun = np.inf

def save_best(x, convergence):
    global best_solution, best_fun
    # 计算当前目标函数值
    current_fun = perturb_para(x, simple_para)
    
    if current_fun < best_fun:
        best_fun = current_fun
        best_solution = x.copy()
        # 自动保存到文件
        np.save("best_result.npy", best_solution)
        print(f"\n✅ 已保存新最优解，fun = {best_fun:.6e}")

if __name__ == "__main__":
    simple_para = pd.read_csv(r"mesh_para\40plus4_bwb.csv").to_numpy()
    base_aircraft = Aircraft(simple_para)

    bounds = [(-0.5, 0.5)] * 10
    x0 = x0 = np.random.rand(10)

    # best_x = np.load("best_result.npy")
    # print("中断前最优解：", best_x)

    result = differential_evolution(
        perturb_para,
        bounds,
        args=(simple_para, ),
        maxiter=1000,
        tol=1e-6,
        workers=1,
        disp=True,
        callback=save_best
    )

    # ---------------------
    # 输出结果
    # ---------------------
    print("优化成功！")
    print("最优参数：", result.x)
    print("最小面积：", result.fun)

    # x = best_x
    # print(perturb_para(x, simple_para))
    # pd.DataFrame(simple_para).to_csv(r"mesh_para\\opt_bwb.csv", index=False)