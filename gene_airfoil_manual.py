from matplotlib.pylab import f
import numpy as np
import scipy
import scipy.interpolate
from scipy.special import comb
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import differential_evolution

def cst_rec(para, N1=0.5, N2=1, n_points=60, psi_end=1.0):
        ##从参数列表提取参数赋值变量
        order = int((len(para) - 8)/2)
        coeffs = np.array([para[1:order+2],para[order+2:(order+1)*2+1]])
        le = para[-5]; te = para[-4]; z_offset = para[-3]; dy_upper = para[-2]; dy_lower = para[-1]

        psi = np.linspace(0, psi_end, n_points)
        coeffs_upper = coeffs[0]
        coeffs_lower = coeffs[1]
        
        # 生成Bernstein基函数
        B = np.zeros((n_points, order+1))
        for i in range(order+1):
            B[:, i] = comb(order, i) * (psi**i) * (1 - psi)**(order-i)
        
        # 计算上下表面坐标
        y_upper = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_upper) + psi*dy_upper
        y_lower = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_lower) + psi*dy_lower

        chord = te - le
        x_true = le + chord*psi
        y_upper = chord*y_upper + z_offset
        y_lower = chord*y_lower + z_offset

        coord_u = np.array([x_true,y_upper]).T
        coord_l = np.array([x_true, y_lower]).T

        return coord_u, coord_l

def fit_cst_airfoil(target_u, target_l, order=4, N1=0.5, N2=1):
    """
    输入：target_u 上表面坐标 (n,2)，target_l 下表面坐标 (n,2)
    输出：para 完整参数数组 → 可直接传入你的 cst_rec()
    """
    x = np.linspace(target_u[0, 0], target_u[-1, 0], 60)
    f = scipy.interpolate.interp1d(target_u[:, 0], target_u[:, 1], kind=4)
    target_u = np.array([x, f(x)]).T
    f = scipy.interpolate.interp1d(target_l[:, 0], target_l[:, 1], kind=6)
    target_l = np.array([x, f(x)]).T
    # ===================== 步骤1：从目标翼型自动提取固定参数 =====================
    x_target = target_u[:, 0]
    le = x_target[0]          # 前缘x
    te = x_target[-1]         # 后缘x
    chord = te - le           # 弦长
    z_offset = target_u[0, 1] # 固定为0
    
    # 后缘dy：直接取后缘点差值
    dy_upper = target_u[-1, 1] / chord
    dy_lower = target_l[-1, 1] / chord

    # 归一化x坐标 psi
    psi = (x_target - le) / chord
    n_points = len(psi)

    # ===================== 步骤2：构建伯恩斯坦基函数 =====================
    B = np.zeros((n_points, order + 1))
    for i in range(order + 1):
        B[:, i] = comb(order, i) * (psi**i) * ((1 - psi)**(order - i))
    shape_fun = (psi**N1) * ((1 - psi)**N2)

    # ===================== 步骤3：拟合残差函数 =====================
    def residual(cst_coeffs):
        # 拆分上下表面系数
        cu = cst_coeffs[:order+1]
        cl = cst_coeffs[order+1:]
        
        # 计算CST形状
        y_u = (shape_fun * (B @ cu) + psi * dy_upper) * chord
        y_l = (shape_fun * (B @ cl) + psi * dy_lower) * chord
        
        # 残差：上下表面同时拟合
        res_u = y_u - target_u[:, 1]
        res_l = y_l - target_l[:, 1]
        return np.concatenate([res_u, res_l])

    # ===================== 步骤4：最小二乘拟合 =====================
    x0 = np.zeros(2 * (order + 1))  # 初始值全0
    res = least_squares(residual, x0, loss='linear')

    # ===================== 步骤5：组装成你函数需要的完整 para =====================
    cst_fit = res.x
    para = np.concatenate([
        [0.0],                # para[0]
        cst_fit,              # CST系数（上+下）
        [le, te, z_offset, dy_upper, dy_lower]  # 固定参数
    ])
    
    return para, res.fun

def cal(cst):
    extra_para = np.array([0., 72.0, 0., 0.0208, 0.0208])
    para = np.concatenate([[0.], cst, extra_para], axis=0)
    coord_u, coord_l = cst_rec(para, n_points=120)
    thick_list = []; area = 0.
    flag = 0; start_cabin = 0.; end_cabin = 0.
    for u, l in zip(coord_u, coord_l):
        thick = u[1]-l[1]
        thick_list.append(thick)
        if thick >= 2.0 and u[0] >= 15 and flag == 0:
            start_cabin = u[0]
            flag = 1
        if flag == 1 and thick < 2.0:
            end_cabin = u[0]
            flag = 2 ##结束客舱判定
        if thick <= 0:
            area += 1
        if u[1] > 1.55:
            area += 10*(u[1]-1.55)
        if l[1] < -0.55:
            area += 10*(-0.55-l[1])
    thick_list = np.array(thick_list)
    cabin_length = end_cabin - start_cabin - 38
    area += thick_list.sum()/60

    return area, cabin_length

def opt(cst):
    area, L = cal(cst)
    if L < 0:
        return area - 1000*L
    return area

if __name__ == "__main__":
    extra_para = [0., 72.0, 0, 0.0208, 0.0208]
    n = 12
    target_u = np.array([[0.0, 0.0]] + [[round(20 + i * (51.04 - 20) / (n-1), 2), 1.55] for i in range(n)] + [[72.0, 1.05]])
    target_l = np.array([[0.0, 0.0], [10.5, 0.45]] + [[round(20 + i * (51.04 - 20) / (n-1), 2), -0.65] for i in range(n)] + [[72.0, 1.05]])

    z_off = 0.5
    sec_2 = np.array([10.0, z_off]); sec_2_end = np.array([70.0, 1.05])
    target_u[0] = sec_2; target_u[-1] = sec_2_end; target_u[1:, 1] -= z_off; target_u[1:-1, 1]
    target_l[0] = sec_2; target_l[-1] = sec_2_end; target_l[1:, 1] -= z_off; target_l[1:-1, 1] += 0.2
    plt.scatter(target_u[:, 0], target_u[:, 1])
    plt.scatter(target_l[:, 0], target_l[:, 1])

    para, res = fit_cst_airfoil(target_u, target_l, 8)

    # constraint = {
    #     'type': 'ineq',    # 不等式约束：fun(x) >= 0
    #     'fun': constraint_fun
    # }
    bounds = [(-0.5, 0.5)] * 6
    x0 = x0 = np.random.rand(6) * 0.1 - 0.05

    # result = differential_evolution(
    #     opt,
    #     bounds,
    #     maxiter=10000,
    #     tol=1e-6,
    #     workers=1,
    #     disp=True
    # )

    # # ---------------------
    # # 输出结果
    # # ---------------------
    # print("优化成功！")
    # print("最优参数：", result.x)
    # print("最小面积：", result.fun)
    # print(cal(result.x))
    # para = np.concatenate([[0.], result.x, extra_para], axis=0)
    print(para)
    coord_u, coord_l = cst_rec(para)
    plt.plot(coord_u[:, 0], coord_u[:, 1])
    plt.plot(coord_l[:, 0], coord_l[:, 1])
    plt.show()