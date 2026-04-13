from aircraft_gene import Aircraft

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

def perturb_para(x, base_para):
    new_para = base_para.copy()
    new_para[1:, -2:] += x.reshape([4, 1])
    new_air = Aircraft(new_para)
    tu, tl = new_air.if_smooth()
    l = new_air.Laplace()
    sm = tu+tl+5*(l[0]+l[1])
    # passenger = new_air.cal_volume()-160
    # if passenger < 0:
    #     sm -= 1000*passenger

    return sm

if __name__ == "__main__":
    simple_para = pd.read_csv(r"mesh_para\165_6.64.csv").to_numpy()
    base_aircraft = Aircraft(simple_para)

    bounds = [(-0.05, 0.05)] * 4
    x0 = x0 = np.random.rand(4)*0.1 - 0.05

    result = differential_evolution(
        perturb_para,
        bounds,
        args=(simple_para, ),
        maxiter=1000,
        tol=1e-6,
        workers=1,
        disp=True
    )

    # ---------------------
    # 输出结果
    # ---------------------
    print("优化成功！")
    print("最优参数：", result.x)
    print("最小面积：", result.fun)

    x = result.x
    simple_para[1:, -2:] += x
    pd.DataFrame(simple_para).to_csv(r"mesh_para\\opt_bwb.csv", index=False)