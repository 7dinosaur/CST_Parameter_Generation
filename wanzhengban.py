import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
import scipy.interpolate as si
from typing import List, Tuple
import os

class Section:
    def __init__(self, y_pos, points):
        self.y_pos = y_pos  # 展向位置
        self.points = points  # 弦向点坐标数组，形状为(N, 2)

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
    print(coord_norm.shape)
    
    return coord_norm, x[le_index], x[te_index], z_offset

def reconstruct_airfoil(coeffs, N, dy_upper=0, dy_lower=0, N1=0.001, N2=0.001, n_points=60):
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

def reconstruct_01(coeffs, N, dy_upper=0, dy_lower=0, N1=0.001, N2=0.001, n_points=60):
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

def reconstruct(coeffs, N, le, te, z_offset, dy_upper=0, dy_lower=0, N1=0.001, N2=0.001, n_points=60, bu0 = False, le0 = 0.0, te0 = 0.0):
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

    if bu0:
        true_psi = np.linspace(le0, te0, n_points)
        flag = 1
        for i in range(n_points):
            if true_psi[i] > le and flag:
                airfoil_le_idx = i - 1
                flag = 0
            if true_psi[i] < te and not flag:
                airfoil_te_idx = i + 1
        airfoil_n_points = airfoil_te_idx - airfoil_le_idx + 1
        airfoil_psi = np.linspace(0, 1, airfoil_n_points)
        B = np.zeros((airfoil_n_points, N+1))
        for i in range(N+1):
            B[:, i] = comb(N, i) * (airfoil_psi**i) * (1 - airfoil_psi)**(N-i)
        
        # 计算上下表面坐标
        y_upper = (airfoil_psi**N1 * (1 - airfoil_psi)**N2) * (B @ coeffs_upper) + airfoil_psi*dy_upper
        y_lower = (airfoil_psi**N1 * (1 - airfoil_psi)**N2) * (B @ coeffs_lower) + airfoil_psi*dy_lower

        chord = true_psi[airfoil_te_idx] - true_psi[airfoil_le_idx]
        
        x_up = le + chord*airfoil_psi
        x_up = np.concatenate((true_psi[:airfoil_le_idx], x_up, true_psi[airfoil_te_idx+1:]))
        x_true = np.append(x_up[::-1], x_up)
        y_up = chord*y_upper + z_offset
        y_up = np.concatenate((np.zeros_like(true_psi[:airfoil_le_idx])+z_offset, y_up, np.zeros_like(true_psi[airfoil_te_idx+1:])+z_offset))
        y_low = chord*y_lower + z_offset
        y_low = np.concatenate((np.zeros_like(true_psi[:airfoil_le_idx])+z_offset, y_low, np.zeros_like(true_psi[airfoil_te_idx+1:])+z_offset))
        y_true = np.append(y_up[::-1], y_low)

        coord_true = np.array([x_true,y_true]).T
        print(coord_true.shape, "最后一个")
        
    else:
        chord = te - le
        x = coords[:,0]
        y = coords[:,1]

        x_true = le + chord*x
        y_true = chord*y + z_offset

        coord_true = np.array([x_true,y_true]).T

    return coord_true

def interpolate_cst3(mesh_para, total_num_span_points_interval: int) -> List[Tuple[float, np.ndarray]]:
    y_list = mesh_para[:,0]
    full_y_list = np.append(-y_list[1:][::-1], y_list)
    print(full_y_list)
    full_para = np.empty(0)
    full_y = np.linspace(y_list[0], y_list[-1], total_num_span_points_interval)
    full_para = full_y.reshape([-1,1])

    for j in range(mesh_para.shape[1]-1):
        full_mesh_para = np.append(mesh_para[:,j+1][1:][::-1], mesh_para[:,j+1])
        
        if j == mesh_para.shape[1]-5:
            f = si.interp1d(full_y_list, full_mesh_para, kind=1)
        else:
            f = si.interp1d(full_y_list, full_mesh_para, kind=2)
            # f = si.Akima1DInterpolator(full_y_list, full_mesh_para)
        # f = si.interp1d(full_y_list, full_mesh_para, kind='cubic')
        para = f(full_y).reshape([-1,1])
        full_para = np.append(full_para, para, axis=1)
    print(full_para.shape)
    return full_para

def interpolate_cst3_nosym(mesh_para, total_num_span_points_interval: int) -> List[Tuple[float, np.ndarray]]:
    # 取消对称：直接使用原始y_list，不再生成对称的full_y_list
    y_list = mesh_para[:,0]
    # 移除对称y坐标的生成逻辑
    # full_y_list = np.append(-y_list[1:][::-1], y_list)  # 注释掉对称逻辑
    # print(full_y_list)  # 对称相关打印也注释
    
    full_para = np.empty(0)
    # 插值的y坐标范围改为原始y_list的首尾（不再是对称范围）
    full_y = np.linspace(y_list[0], y_list[-1], total_num_span_points_interval)
    full_para = full_y.reshape([-1,1])

    for j in range(mesh_para.shape[1]-1):
        # 取消对称：直接使用原始的mesh_para[:,j+1]，不再生成对称的full_mesh_para
        full_mesh_para = mesh_para[:,j+1]  # 移除对称参数的生成逻辑
        
        # 保留原有的插值类型判断逻辑
        if j == mesh_para.shape[1]-5:
            f = si.interp1d(y_list, full_mesh_para, kind=1)  # 插值x轴改为原始y_list
        else:
            f = si.interp1d(y_list, full_mesh_para, kind=2)  # 插值x轴改为原始y_list
            # f = si.Akima1DInterpolator(y_list, full_mesh_para)
        
        # 保留原有的插值和拼接逻辑
        para = f(full_y).reshape([-1,1])
        full_para = np.append(full_para, para, axis=1)
    
    print(full_para.shape)
    return full_para


def generate_3d_mesh(
    interp_sections: List[Tuple[float, np.ndarray]],
    output_file: str = "wing_mesh.dat"
) -> None:
    def write_xyz(coords, f):
        x_flat = coords[:, :, 0].flatten()
        for i in range(0, len(x_flat), 4):
            f.write(" ".join(f"{x:.6f}" for x in x_flat[i:i+4]) + "\n")
        f.write("\n")
        
        # Y坐标块
        y_flat = coords[:, :, 1].flatten()
        for i in range(0, len(y_flat), 4):
            f.write(" ".join(f"{y:.6f}" for y in y_flat[i:i+4]) + "\n")
        f.write("\n")
        
        # Z坐标块
        z_flat = coords[:, :, 2].flatten()
        for i in range(0, len(z_flat), 4):
            f.write(" ".join(f"{z:.6f}" for z in z_flat[i:i+4]) + "\n")

    n_sec = len(interp_sections)
    print(n_sec)
    """生成三维网格点并格式化输出"""
    upper_sections = []
    lower_sections = []
    for i in range(n_sec):
        y = interp_sections[i][0]
        coords = interp_sections[i][1]
        n = int(len(coords)/2)
        upper_sections.append((y, coords[:n]))
        lower_sections.append((y, coords[n:]))

    with open(output_file, 'w') as f:
        # 存储所有网格点
        all_points = []
        f.write("3\n")
        f.write(f"{n_sec} {n} 1\n")
        f.write(f"{n_sec} {n} 1\n")
        f.write(f"{2} {n} 1\n")
        
        # 遍历每个插值剖面
        for y, section in upper_sections:
            # 添加当前剖面的所有点[x, y, z]
            # 注意：section[:,0]是x坐标，section[:,1]是z坐标
            profile_points = np.column_stack((
                section[:, 0],  # x坐标
                np.full(section.shape[0], y),  # y位置（展向）
                section[:, 1]   # z坐标
            ))
            all_points.append(profile_points)
        
        # 转换为三维数组 [y_layer, point_index, xyz]
        all_points = np.array(all_points)
        
        write_xyz(all_points[:, ::-1].swapaxes(0, 1), f)
        
        all_points = []

        for y, section in lower_sections:
            # 添加当前剖面的所有点[x, y, z]
            # 注意：section[:,0]是x坐标，section[:,1]是z坐标
            profile_points = np.column_stack((
                section[:, 0],  # x坐标
                np.full(section.shape[0], y),  # y位置（展向）
                section[:, 1]   # z坐标
            ))
            all_points.append(profile_points)
        
        # 转换为三维数组 [y_layer, point_index, xyz]
        all_points = np.array(all_points)

        write_xyz(all_points[::-1, :].swapaxes(0, 1), f)

        all_points = []
        tail_up = np.column_stack((upper_sections[-1][1][:,0][::-1], np.full(section.shape[0], upper_sections[-1][0]), upper_sections[-1][1][:,1][::-1]))
        tail_lower = np.column_stack((lower_sections[-1][1][:,0], np.full(section.shape[0], lower_sections[-1][0]), lower_sections[-1][1][:,1]))
        print(tail_up.shape)
        print(lower_sections[-1][1][:,0])
        all_points.append(tail_up)
        all_points.append(tail_lower)
        all_points = np.array(all_points)
        print(all_points.shape)
        write_xyz(all_points.swapaxes(0, 1), f)


    print(f"三维机翼网格已保存至 {output_file}")
    

def export_airfoil_profiles(mesh_para, cst_order, output_dir="airfoil_profiles", n_points=60):
    """
    将原始参数网格中的每个剖面翼型导出为Selig格式文件。

    参数:
        mesh_para (np.ndarray): 包含各剖面CST参数和几何参数的数组。
        cst_order (int): CST参数化的阶数。
        output_dir (str): 输出目录的名称。默认为 "airfoil_profiles"。
        n_points (int): 每个翼型重构时使用的点数。默认为60。

    返回:
        int: 成功导出的翼型文件数量。
    """
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    exported_count = 0

    print(f"正在导出原始参数剖面的翼型文件到目录 '{output_dir}'...")
    
    # 2. 遍历mesh_para的每一行（每个原始剖面）
    for idx, data_line in enumerate(mesh_para):
        try:
            # 3. 提取当前剖面的CST参数和其他几何参数
            # 假设data_line的结构为: [y_pos, A0_upper, A1_upper, ..., A0_lower, A1_lower, ..., le, te, z_offset, dy_upper, dy_lower]
            cst = np.array([data_line[1:cst_order+2], data_line[cst_order+2:(cst_order+1)*2+1]])
            dy_upper = data_line[-2] # 可能的上表面偏移
            dy_lower = data_line[-1] # 可能的下表面偏移

            # 4. 重构翼型坐标（归一化坐标）
            coords_normalized = reconstruct_01(cst, cst_order, dy_upper, dy_lower, N1=0.5, N2=1, n_points=n_points)

            # 5. 调整坐标顺序以满足Selig格式 (从后缘开始，顺时针或逆时针闭合)
            n_points_total = len(coords_normalized)
            n_half = n_points_total // 2
            upper_surface = coords_normalized[:n_half]   # 上表面点
            lower_surface = coords_normalized[n_half:]    # 下表面点
            lower_surface_ordered = lower_surface[::-1]   # 反转下表面顺序
            upper_surface_ordered = upper_surface[::-1]   # 反转上表面顺序
            selig_coords = np.vstack((lower_surface_ordered, upper_surface_ordered))
            if idx == 0:    
                plt.plot(selig_coords[:,0], selig_coords[:,1])
                plt.ylim(-0.5,0.5)
                plt.xlim(0,1)

            # 6. 定义输出文件名和路径
            filename = f"{idx}.dat" # 使用自然数索引命名
            filepath = os.path.join(output_dir, filename)

            # 7. 写入文件
            with open(filepath, 'w') as f:
                # 可选：添加一行注释头
                # f.write(f"# Airfoil profile from original section {idx}\n")
                for point in selig_coords:
                    f.write(f"{point[0]:.6f}   {point[1]:.6f}\n")

            exported_count += 1
            # print(f"已导出: {filepath}") # 如需详细日志可取消注释

        except Exception as e:
            print(f"处理剖面索引 {idx} 时出错: {e}")
            # 可以选择继续处理下一个剖面或抛出异常

    print(f"所有原始剖面翼型文件导出完毕。共成功导出 {exported_count} 个文件。")
    return exported_count

if __name__ == "__main__":
    mesh_para = pd.read_csv(r"mesh_para\increase_cabin.csv").to_numpy()
    cst_order = 8
    N1 = 0.5
    N2 = 1
    export_airfoil_profiles(mesh_para, cst_order, output_dir="airfoil_profiles_original")
    sections = []
    # interpolated_cst = interpolate_cst(mesh_para, 3)
    interpolated_cst = interpolate_cst3(mesh_para, 161)
    # interpolated_cst = mesh_para
    for data in interpolated_cst:
        cst = np.array([data[1:cst_order+2],data[cst_order+2:(cst_order+1)*2+1]])
        # print(cst)
        # coords = reconstruct_airfoil(cst, cst_order, data[-2], data[-1], N1, N2)
        # coords = reconstruct_truefile(coords, data[-5], data[-4], data[-3])
        coords = reconstruct(cst, cst_order, data[-5], data[-4], data[-3], data[-2], data[-1], N1, N2, 261)
        # section.append(coords)
        sections.append((data[0], coords))
    generate_3d_mesh(sections, r"geo\increase_cabin.x")
    plt.show()