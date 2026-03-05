import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
import os


def process_airfoil_coordinates(data):
    x = data[:,0]
    z = data[:,1]
    # 1. 计算前缘点（x最小点）和后缘点（x最大点）
    le_index = np.argmin(x)  # 前缘索引
    te_index = np.argmax(x)  # 后缘索引

    # 2. 计算z方向整体偏移量（取前缘点z值）
    z_offset = z[le_index]  # 前缘点z值应为0，但原始数据可能偏移

    # 3. 平移所有点：消除z偏移
    z_centered = z - z_offset  # 使前缘点z=0

    # 4. 计算弦长并缩放
    chord = x[te_index] - x[le_index]  # 弦长 = 后缘x - 前缘x
    x_norm = (x - x[le_index]) / chord  # 前缘x→0, 后缘x→1
    z_norm = z_centered / chord         # 保持比例缩放

    coord_norm = np.array([x_norm,z_norm]).T
    # print(coord_norm.shape)
    
    return coord_norm, x[le_index], x[te_index], z_offset

def read_coords(data):
    x = data[:,0]
    z = data[:,1]
    # 1. 计算前缘点（x最小点）和后缘点（x最大点）
    le_index = np.argmin(x)  # 前缘索引
    te_index = np.argmax(x)  # 后缘索引

    co_up = []
    co_low = []
    # 2. 计算z方向整体偏移量（取前缘点z值）
    z_offset = z[le_index]  # 前缘点z值应为0，但原始数据可能偏移
    for da in data:
        x_co = da[0]
        z_co = da[1]
        z_biao = (x_co - x[le_index]) * (z[te_index] - z[le_index])/(x[te_index] - x[le_index]) + z[le_index]
        if z_co > z_biao:
            co_up.append(da)
        else:
            co_low.append(da)

    co_up = np.array(co_up)
    co_low = np.array(co_low)
    plt.plot(co_up[:,0], co_up[:,1])
    plt.plot(co_low[:,0], co_low[:,1])
    plt.show()
    print(co_up, co_low)

    # 3. 平移所有点：消除z偏移
    z_centered = z - z_offset  # 使前缘点z=0

    # 4. 计算弦长并缩放
    chord = x[te_index] - x[le_index]  # 弦长 = 后缘x - 前缘x
    x_norm = (x - x[le_index]) / chord  # 前缘x→0, 后缘x→1
    z_norm = z_centered / chord         # 保持比例缩放

    coord_norm = np.array([x_norm,z_norm]).T
    # print(coord_norm.shape)
    
    return coord_norm, x[le_index], x[te_index], z_offset

def cst_fitting(coords, N, N1=0.5, N2=1.0):
    """CST参数拟合核心函数"""

    def fit(x, y, dy_te):
    
        # 类函数参数（N1=0.5，N2=1.0）
        C = (x**N1) * (1 - x)**N2
        
        # 构造Bernstein基函数矩阵
        A = np.zeros((len(x), N+1))
        for i in range(N+1):
            A[:, i] = comb(N, i) * (x**i) * (1 - x)**(N-i)

        y_adjusted = y - x * dy_te
        # 最小二乘拟合
        coeffs = np.linalg.lstsq(A * C[:, np.newaxis], y_adjusted, rcond=None)[0]
        return coeffs

    num_co = len(coords)
    coords_upper = coords[:int(num_co/2)]  # 上表面（包含前缘点）
    coords_lower = coords[int(num_co/2):]     # 下表面（包含前缘点）
    dy_upper = coords[0,1]
    dy_lower = coords[-1,1]
    cst_upper = fit(coords_upper[:,0], coords_upper[:,1], dy_upper)
    cst_lower = fit(coords_lower[:,0], coords_lower[:,1], dy_lower)
    
    return np.array([cst_upper,cst_lower]), dy_upper, dy_lower

def reconstruct_airfoil(coeffs, N, dy_upper=0, dy_lower=0, N1=0.001, N2=0.001, n_points=100):
    """重构CST翼型"""
    psi = np.linspace(0, 1, n_points)
    coeffs_upper = coeffs[0]
    coeffs_lower = coeffs[1]
    
    # 生成Bernstein基函数
    B = np.zeros((n_points, N+1))
    for i in range(N+1):
        B[:, i] = comb(N, i) * (psi**i) * (1 - psi)**(N-i)
    
    # 计算上下表面坐标
    y_upper = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_upper) + psi*dy_upper
    y_lower = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_lower) + psi*dy_lower
    x = np.append(psi[::-1], psi)
    y = np.append(y_upper[::-1], y_lower)
    coords = np.array([x,y]).T
    
    return coords

def reconstruct_truefile(coords, le, te, z_offset):
    chord = te - le
    x = coords[:,0]
    y = coords[:,1]

    x_true = le + chord*x
    y_true = chord*y + z_offset

    coord_true = np.array([x_true,y_true]).T

    return coord_true

def flatten(data):
     return np.array(data).flatten()

if __name__ == "__main__":
    folder_path = r"increase_cabin_sec"
    file_names = os.listdir(folder_path)
    print(file_names)
    y_list = []
    for filename in file_names:
        num_str = filename.replace("y=", "").replace(".dat", "")
        try:
            y_list.append(float(num_str))
        except ValueError:
            pass  # 处理格式错误
    y_list = np.sort(y_list)
    print(y_list)
    mesh_para = []
    for y in y_list:
        data = np.loadtxt(f'{folder_path}\\y={y}.dat')
        cst_order = 8
        N1 = 0.5
        N2 = 1
        coord_norm, le, te, z_offset = process_airfoil_coordinates(data)
        # coord_norm, le, te, z_offset = read_coords(data)
        coeffs, dy_upper, dy_lower = cst_fitting(coord_norm, cst_order, N1, N2)
        coords = reconstruct_airfoil(coeffs, cst_order, dy_upper, dy_lower, N1, N2)

        plt.plot(coords[:, 0], coords[:, 1])
        para = np.concatenate((flatten(y), flatten(coeffs), flatten(le), flatten(te), flatten(z_offset), flatten(dy_upper), flatten(dy_lower)))
        mesh_para.append(para)
        # print(y, coeffs, le, te, z_offset, dy_upper, dy_lower)
    mesh_para = np.array(mesh_para)
    csv_data = pd.DataFrame(mesh_para)
    csv_data.to_csv(r"increase_cabin.csv", index=None)
    mesh_para = pd.read_csv(r"increase_cabin.csv").to_numpy()
    print(mesh_para)
    plt.show()