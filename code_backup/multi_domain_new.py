import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
import scipy.interpolate as si
from typing import List, Tuple
from numpy.typing import NDArray

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
    # print(coord_norm.shape)
    
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

def reconstruct_truefile(coords, le, te, z_offset, bu0 = False, le0 = 0.0, te0 = 0.0, n_points = 60):
    if bu0:
        true_psi = np.linspace(te0, le0, n_points)
        flag = 1
        for i in range(n_points):
            if true_psi[i] > le and flag:
                airfoil_le_idx = i - 1
                flag = 0
            if true_psi[i] < te and not flag:
                airfoil_te_idx = i + 1
        airfoil_n_point = airfoil_te_idx - airfoil_le_idx + 1
        coords = 1
    else:
        chord = te - le
        x = coords[:,0]
        y = coords[:,1]

        x_true = le + chord*x
        y_true = chord*y + z_offset

        coord_true = np.array([x_true,y_true]).T

    return coord_true

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
        # print(coord_true.shape)

    return coord_true

def interpolate_sections(sections: List[Section], num_span_points_per_interval: int) -> List[Tuple[float, np.ndarray]]:
    """
    Returns:
        List[Tuple[float, np.ndarray]]: 插值后的剖面列表，每个元素为(y位置, 弦向点坐标数组)
    """
    if not sections:
        raise ValueError("剖面列表不能为空")
    if num_span_points_per_interval < 1:
        raise ValueError("相邻剖面间插值点数量至少为1")
    
    # 按y_pos排序剖面
    sorted_sections = sorted(sections, key=lambda s: s.y_pos)
    
    
    result = []
    # 遍历每一对相邻剖面
    for i in range(len(sorted_sections) - 1):
        y_start = sorted_sections[i].y_pos
        y_end = sorted_sections[i+1].y_pos
        
        # 生成当前区间内的插值y位置（包含左端点，不包含右端点）
        y_interval = np.linspace(y_start, y_end, num_span_points_per_interval + 1)[:-1]
        
        # 对每个插值位置y，生成插值后的弦向点
        for y in y_interval:
            interp_profile = []
            # 遍历每个弦向点索引
            for j in range(len(sorted_sections[0].points)):
                # 提取当前弦向点在两个相邻剖面的坐标
                x0 = sorted_sections[i].points[j, 0]
                z0 = sorted_sections[i].points[j, 1]
                x1 = sorted_sections[i+1].points[j, 0]
                z1 = sorted_sections[i+1].points[j, 1]
                
                # 计算插值比例因子（基于y位置）
                t = (y - y_start) / (y_end - y_start)
                
                # 线性插值当前弦向点
                x_interp = x0 + t * (x1 - x0)
                z_interp = z0 + t * (z1 - z0)
                interp_profile.append([x_interp, z_interp])
            
            result.append((y, np.array(interp_profile)))
    
    # 添加最后一个原始剖面（右端点）
    last_section = sorted_sections[-1]
    result.append((last_section.y_pos, last_section.points.copy()))
    
    return result

def interpolate_cst(mesh_para, num_span_points_per_interval: int) -> List[Tuple[float, np.ndarray]]:
    y_list = mesh_para[:,0]
    inter_para = []
    full_para = []
    for i in range(len(y_list)-1):
        # 生成当前区间内的插值y位置（包含左端点，不包含右端点）
        y_interval = np.linspace(y_list[i], y_list[i+1], num_span_points_per_interval + 1)
        full_para.append(mesh_para[i])
        for y in y_interval[1:-1]:
            inter_y = [y]
            for j in range(23):
                para = np.interp(y, y_list, mesh_para[:,j+1])
                inter_y.append(para)
            inter_para.append(inter_y)
            full_para.append(inter_y)
    full_para.append(mesh_para[-1])
    inter_para = np.array(inter_para)
    full_para = np.array(full_para)
    print(full_para.shape)
    return full_para

def interpolate_cst2(mesh_para, num_span_points_per_interval: int) -> List[Tuple[float, np.ndarray]]:
    y_list = mesh_para[:,0]
    full_para = np.empty(0)
    full_y = np.empty(0)
    for i in range(len(y_list)-1):
        # 生成当前区间内的插值y位置（包含左端点，不包含右端点）
        y_interval = np.linspace(y_list[i], y_list[i+1], num_span_points_per_interval + 1)
        full_y = np.append(full_y, y_interval[:-1])
    full_y = np.append(full_y, y_list[-1])
    full_para = full_y.reshape([-1,1])
    for j in range(31):
        if j == 19:
            f = si.interp1d(y_list, mesh_para[:,j+1])
        else:
            f = si.interp1d(y_list, mesh_para[:,j+1], kind='quadratic')
        # f = si.interp1d(y_list, mesh_para[:,j+1], kind='quadratic')
        para = f(full_y).reshape([-1,1])
        full_para = np.append(full_para, para, axis=1)
    print(full_para.shape)
    return full_para

def interpolate_cst3(mesh_para, total_num_span_points_interval: int) -> NDArray[np.float64]:
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
        # f = si.interp1d(y_list, mesh_para[:,j+1], kind='quadratic')
        para = f(full_y).reshape([-1,1])
        full_para = np.append(full_para, para, axis=1)
    print(full_para.shape)
    return full_para

def interpolate_cst_single(mesh_para, y):
    y_list = mesh_para[:,0]
    full_y_list = np.append(-y_list[1:][::-1], y_list)
    # print(full_y_list)
    single_para = np.array([y])

    for j in range(mesh_para.shape[1]-1):
        full_mesh_para = np.append(mesh_para[:,j+1][1:][::-1], mesh_para[:,j+1])
        
        if j == mesh_para.shape[1]-5:
            f = si.interp1d(full_y_list, full_mesh_para, kind=1)
        else:
            f = si.interp1d(full_y_list, full_mesh_para, kind=2)
        # f = si.interp1d(y_list, mesh_para[:,j+1], kind='quadratic')
        para = f(y)
        single_para = np.append(single_para, para)

    return single_para
                
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

def generate_3d_mesh(
    interp_sections: List[Tuple[float, np.ndarray]],
    output_file: str = "wing_mesh.dat"
) -> None:

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

def generate_multi_domain(mesh_para, output_file: str = "wing_mesh.dat") -> None:
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

    print(mesh_para[12])
    n_sec = len(mesh_para)
    with open(output_file, 'w') as f:
        # 存储所有网格点
        all_points = []
        f.write("1\n")
        f.write(f"{n_sec}  1\n")

def reconstruct(psi, coeffs, N, le, te, z_offset, dy_upper=0, dy_lower=0, N1=0.5, N2=1, n_points=60):

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

    chord = te - le
    x = coords[:,0]
    y = coords[:,1]

    x_true = le + chord*x
    y_true = chord*y + z_offset

    coord_true = np.array([x_true,y_true]).T
    # print(coord_true.shape)

    return coord_true

def cst_re(psi, coeffs, N, dy_upper=0, dy_lower=0, N1=0.5, N2=1, n_points=60):

    coeffs_upper = coeffs[0]
    coeffs_lower = coeffs[1]
    
    # 生成Bernstein基函数
    B = np.zeros((n_points, N+1))
    for i in range(N+1):
        B[:, i] = comb(N, i) * (psi**i) * (1 - psi)**(N-i)
    
    # 计算上下表面坐标
    y_upper = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_upper) + psi*dy_upper
    y_lower = (psi**N1 * (1 - psi)**N2) * (B @ coeffs_lower) + psi*dy_lower

    coords = np.array([psi,y_upper]).T

    return coords

def cst_single(psi, coeffs, N, dy_upper, N1=0.5, N2=1, n_points=1):
    
    # 生成Bernstein基函数
    B = np.zeros((n_points, N+1))
    for i in range(N+1):
        B[:, i] = comb(N, i) * (psi**i) * (1 - psi)**(N-i)
    
    # 计算上下表面坐标
    y_upper = (psi**N1 * (1 - psi)**N2) * (B @ coeffs) + psi*dy_upper
    y_upper = y_upper[0]

    return y_upper

def mesh_test(
    all_points,
    output_file: str = "wing_mesh.dat"
) -> None:

    n_dom = len(all_points)

    with open(output_file, 'w') as f:
        # 存储所有网格点
        f.write(f"{n_dom}\n")
        for dom in all_points:
            f.write(f"{dom.shape[1]} {dom.shape[0]} 1\n")

        for dom in all_points:
            write_xyz(dom, f)

def redistribution(x, y, n):
    xtmp = np.linspace(0, 1, 100)
    x_tmp = x[0] + xtmp*x[-1]
    x = np.append(-np.array(x)[1:][::-1], np.array(x))
    y = np.append(np.array(y)[1:][::-1], np.array(y))

    ##固定末尾导数处理
    # x_mo = x[-1] + 0.0001
    # y_mo = y[-1] - 0.01
    # x = np.append(-x_mo, np.append(x, x_mo))
    # y = np.append(y_mo, np.append(y, y_mo))
    ##——————————————##

    # plt.plot(x, y, marker='o')
    # if len(x) < 4:
    #     fx =si.interp1d(x, y, kind=2)
    # else:
    #     fx =si.interp1d(x, y, kind=2)
    # print(x.shape, y.shape)
    fx = si.Akima1DInterpolator(x, y)


    y_tmp = fx(x_tmp)
    # plt.plot(x_tmp, y_tmp)
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    # plt.show()
    L = np.zeros(100)
    for i in range(100 - 1):
        dx = x_tmp[i+1] - x_tmp[i]
        dy = y_tmp[i+1] - y_tmp[i]
        L[i+1] = (dx**2 + dy**2)**0.5 + L[i]
    l_total = L[-1]
    L_inv = si.interp1d(L, x_tmp, kind=2)
    dL = l_total/(n-1)
    L_pingjun = []
    for i in range(n):
        L_pingjun.append(dL*i)
    x_pingjun = np.append(L_inv(L_pingjun[:-1]), x[-1])
    y_pingjun = fx(x_pingjun)

    return x_pingjun, y_pingjun

def redistribution2(x, y, n):
    
    print(x, y)
    plt.plot(y, x, marker='o')
    # plt.show()
    fy =si.CubicSpline(y[::-1], x[::-1])
    ytmp = np.linspace(y[0], y[int(len(y)/2)], 100)
    xtmp = fy(ytmp)
    ytmp2 = np.linspace(y[0], y[-1], 100)
    xtmp2 = fy(ytmp2)
    plt.plot(ytmp2, xtmp2)

    L = np.zeros(100)
    for i in range(100 - 1):
        dx = xtmp[i+1] - xtmp[i]
        dy = ytmp[i+1] - ytmp[i]
        L[i+1] = (dx**2 + dy**2)**0.5 + L[i]
    l_total = L[-1]
    L_inv = si.interp1d(L, ytmp, kind=2)
    dL = l_total/(n-1)
    L_pingjun = []
    for i in range(n):
        L_pingjun.append(dL*i)
    y_pingjun = np.append(L_inv(L_pingjun[:-1]), y[int(len(y)/2)])
    x_pingjun = fy(y_pingjun)
    plt.plot(y_pingjun, x_pingjun)
    plt.show()

    return x_pingjun, y_pingjun

def mesh_fenkuai(mesh_para):
    cst_order = int((mesh_para.shape[1]-8)/2)
    N1 = 0.5
    N2 = 1
    
    dom12_end = 17.67#第一块网格位置
    # dom34_end = 67.56#第二块网格位置
    i_dom12 = 17
    j_dom12 = 80#机头网格数量
    i_dom34 = i_dom12
    j_dom34 = 256
    i_dom67 = 121
    j_dom67 = j_dom34
    i_dom910 = i_dom34
    j_dom910 = 15
    print("输入参数：", mesh_para.shape)
    interpolated_cst = interpolate_cst3(mesh_para, 61)
    print("插值后参数：", interpolated_cst.shape)

    data0 = mesh_para[0]
    chord = data0[-4] - data0[-5]
    psi_dom12 = np.linspace(0, (dom12_end - data0[-5])/chord, 100)#计算曲线长度
    cst = np.array([data0[1:cst_order+2],data0[cst_order+2:(cst_order+1)*2+1]])#提取第一个剖面的cst参数
    cst_cood_dom12 = cst_re(psi_dom12, cst, cst_order, data0[-2], data0[-1], N1, N2, 100)
    L12 = np.zeros([100])

    for i in range(100 - 1):
        dx = cst_cood_dom12[i+1, 0] - cst_cood_dom12[i, 0]
        dy = cst_cood_dom12[i+1, 1] - cst_cood_dom12[i, 1]
        L12[i+1] = (dx**2 + dy**2)**0.5 + L12[i]#dom12流向长度计算

    l12_total = L12[-1]
    fx12 = si.interp1d(L12, psi_dom12, kind=2)
    L12_pingjun = []
    dL12 = l12_total/(j_dom12 - 1)
    for i in range(j_dom12):
        L12_pingjun.append(dL12*i)
    L12_pingjun[0] = L12[0]
    L12_pingjun[-1] = L12[-1]
    psi12_pingjun = fx12(L12_pingjun)#dom12新的x坐标序列

    """dom12边缘设定"""
    coords_pingjunt = reconstruct(psi12_pingjun, cst, cst_order, data0[-5], data0[-4], data0[-3], data0[-2], data0[-1], N1, N2, j_dom12)
    coords_pingjun = coords_pingjunt[:j_dom12, :][::-1, :]
    coords_pingjun_low = coords_pingjunt[j_dom12:, :]
    coords_y = np.zeros_like(coords_pingjun[:,0])
    coords_up = np.concatenate((coords_pingjun[:,0].reshape([-1,1]), coords_y.reshape([-1,1]), coords_pingjun[:,1].reshape([-1,1])), axis=1)
    coords_low = np.concatenate((coords_pingjun_low[:,0].reshape([-1,1]), coords_y.reshape([-1,1]), coords_pingjun_low[:,1].reshape([-1,1])), axis=1)

    x_pingjun = interpolated_cst[0, -5] + psi12_pingjun*chord
    qianyuan_x = interpolated_cst[:, -5]
    qianyuan_y = interpolated_cst[:, 0]
    qianyuan_z = interpolated_cst[:, -3]
    qianyuan_y = si.interp1d(qianyuan_x, qianyuan_y, kind=2)(x_pingjun)
    qianyuan_z = si.interp1d(qianyuan_x, qianyuan_z, kind=2)(x_pingjun)
    qianyuan_x = x_pingjun
    qianyuan = np.concatenate((qianyuan_x.reshape([-1,1]), qianyuan_y.reshape([-1,1]), qianyuan_z.reshape([-1,1])), axis=1)
    """dom12边缘设定"""


    dom_1 = np.zeros([i_dom12, j_dom12, 3])
    dom_2 = np.zeros([i_dom12, j_dom12, 3])
    for j in range(j_dom12):
        y_list = np.array([0])
        z_up_list = np.array([coords_up[j, 2]])
        z_low_list = np.array([coords_low[j, 2]])
        for da in interpolated_cst[1:]:
            if da[-5] < x_pingjun[j] - 0.1:
                y_list = np.append(y_list, da[0])
                cst_up = da[1:cst_order+2]
                cst_low = da[cst_order+2:(cst_order+1)*2+1]
                chord_da = da[-4] - da[-5]
                x01 = (coords_up[j, 0] - da[-5])/(chord_da)
                z_up = cst_single(x01, cst_up, cst_order, da[-2])*chord_da + da[-3]
                z_low = cst_single(x01, cst_low, cst_order, da[-1])*chord_da + da[-3]
                z_up_list = np.append(z_up_list, z_up)
                z_low_list = np.append(z_low_list, z_low)
        y_list = np.append(y_list, qianyuan[j, 1])
        z_up_list = np.append(z_up_list, qianyuan[j, 2])
        z_low_list = np.append(z_low_list, qianyuan[j, 2])
        # y_list = np.append(y_list, y_list[:-1][::-1])
        # z_list = np.append(z_list, z_low_list[::-1])
        # z_list = np.append(z_list, coords_low[j, 2])
        if np.linalg.norm(y_list[0] - y_list[1]) < 1e-10:
            y_up_new = np.ones(i_dom12)*y_list[0]
            y_low_new = np.ones(i_dom12)*y_list[0]
            z_up_new = np.ones(i_dom12)*z_up_list[0]
            z_low_new = np.ones(i_dom12)*z_low_list[0]
        else:
            y_up_new, z_up_new = redistribution(y_list, z_up_list, i_dom12)
            y_low_new, z_low_new = redistribution(y_list, z_low_list, i_dom12)
        for i in range(i_dom12):
            dom_1[i, j] = np.array([x_pingjun[j], y_up_new[i], z_up_new[i]])
            dom_2[i, j] = np.array([x_pingjun[j], y_low_new[i], z_low_new[i]])
    print(dom_1.shape)
    print(dom_2.shape)
    ##dom1.dom2

    y_end = dom_1[-1, -1, 1]
    print(dom_1[-1, -1, :], "dom12结束位置")
    data_dom34 = interpolate_cst_single(mesh_para, y_end)
    print("dom34边缘参数：", data_dom34, data_dom34[-4], data_dom34[-5])
    chord_34 = data_dom34[-4] - data_dom34[-5]
    cst34 = np.array([data_dom34[1:cst_order+2],data_dom34[cst_order+2:(cst_order+1)*2+1]])#提取第一个剖面的cst参数
    dom34_end = data_dom34[-4]
    psi_dom34 = np.linspace((dom12_end-data0[-5])/chord, (dom34_end-data0[-5])/chord, 100)#计算曲线长度
    # print("psi_dom34:", psi_dom34)
    cst_cood_dom34 = cst_re(psi_dom34, cst, cst_order, data0[-2], data0[-1], N1, N2, 100)
    L34 = np.zeros([100])

    for i in range(100 - 1):
        dx = cst_cood_dom34[i+1, 0] - cst_cood_dom34[i, 0]
        dy = cst_cood_dom34[i+1, 1] - cst_cood_dom34[i, 1]
        L34[i+1] = (dx**2 + dy**2)**0.5 + L34[i]#dom34流向长度计算

    l34_total = L34[-1]
    # print("L34", L34)
    fx34 = si.interp1d(L34, psi_dom34, kind=2)
    L34_pingjun = []
    dL34 = l34_total/(j_dom34 - 1)
    for i in range(j_dom34):
        L34_pingjun.append(dL34*i)
    L34_pingjun[0] = L34[0]
    L34_pingjun[-1] = L34[-1]
    psi34_pingjun = fx34(L34_pingjun)#dom34新的x坐标序列

    """dom34边缘设定"""
    # print("psi34_pingjun",psi34_pingjun)
    coords34_pingjunt = reconstruct(psi34_pingjun, cst, cst_order, data0[-5], data0[-4], data0[-3], data0[-2], data0[-1], N1, N2, j_dom34)
    coords34_pingjun = coords34_pingjunt[:j_dom34, :][::-1, :]
    coords34_pingjun_low = coords34_pingjunt[j_dom34:, :]
    coords34_y = np.zeros_like(coords34_pingjun[:,0])
    coords34_up = np.concatenate((coords34_pingjun[:,0].reshape([-1,1]), coords34_y.reshape([-1,1]), coords34_pingjun[:,1].reshape([-1,1])), axis=1)
    coords34_low = np.concatenate((coords34_pingjun_low[:,0].reshape([-1,1]), coords34_y.reshape([-1,1]), coords34_pingjun_low[:,1].reshape([-1,1])), axis=1)
    
    
    # print(f"边缘参数：{data_dom34}")
    cst34 = np.array([data_dom34[1:cst_order+2],data_dom34[cst_order+2:(cst_order+1)*2+1]])
    print(coords34_up[:, 0], data_dom34[-5], data_dom34[-4])
    psi34_rp = (coords34_up[:, 0] - data_dom34[-5])/(data_dom34[-4] - data_dom34[-5])
    psi34_rp[0] = 0.0

    # print(coords34_up[:, 0])  ##检查psi34_rp范围
    coords34_rpt = reconstruct(psi34_rp, cst34, cst_order, data_dom34[-5], data_dom34[-4], data_dom34[-3], data_dom34[-2], data_dom34[-1], N1, N2, j_dom34)
    coords34_rp_up = coords34_rpt[:j_dom34, :][::-1, :]
    coords34_rp_low = coords34_rpt[j_dom34:, :]
    coords34_ry = np.ones_like(coords34_rp_up[:,0])*y_end
    coords34_rup = np.concatenate((coords34_rp_up[:,0].reshape([-1,1]), coords34_ry.reshape([-1,1]), coords34_rp_up[:,1].reshape([-1,1])), axis=1)
    coords34_rlow = np.concatenate((coords34_rp_low[:,0].reshape([-1,1]), coords34_ry.reshape([-1,1]), coords34_rp_low[:,1].reshape([-1,1])), axis=1)
    # print(data_dom34, "dom34边缘y")
    """dom34边缘设定"""

    dom_3 = np.zeros([i_dom34, j_dom34, 3])
    dom_4 = np.zeros([i_dom34, j_dom34, 3])
    for j in range(j_dom34):
        y_list = np.array([0])
        z_up_list = np.array([coords34_up[j, 2]])
        z_low_list = np.array([coords34_low[j, 2]])
        """未修改"""
        for da in interpolated_cst[1:]:
            if da[0] < y_end-0.1:
                y_list = np.append(y_list, da[0])
                cst_up = da[1:cst_order+2]
                cst_low = da[cst_order+2:(cst_order+1)*2+1]
                chord_da = da[-4] - da[-5]
                x01 = (coords34_up[j, 0] - da[-5])/(chord_da)
                z_up = cst_single(x01, cst_up, cst_order, da[-2])*chord_da + da[-3]
                z_low = cst_single(x01, cst_low, cst_order, da[-1])*chord_da + da[-3]
                z_up_list = np.append(z_up_list, z_up)
                z_low_list = np.append(z_low_list, z_low)
        y_list = np.append(y_list, coords34_rup[j, 1])
        z_up_list = np.append(z_up_list, coords34_rup[j, 2])
        z_low_list = np.append(z_low_list, coords34_rlow[j, 2])
        plt.plot(y_list, z_up_list)

        # print(z_up_list)
        y_up_new, z_up_new = redistribution(y_list, z_up_list, i_dom34)
        y_low_new, z_low_new = redistribution(y_list, z_low_list, i_dom34)

        for i in range(i_dom34):
            dom_3[i, j] = np.array([coords34_up[j, 0], y_up_new[i], z_up_new[i]])
            dom_4[i, j] = np.array([coords34_up[j, 0], y_low_new[i], z_low_new[i]])
    ##网格封闭性修正
    dom_3[:, 0] = dom_1[:, -1]
    dom_4[:, 0] = dom_2[:, -1]

    # y_end_dom5 = dom_1[-1, -1, 1]
    # data_dom5 = interpolate_cst_single(mesh_para, y_end_dom5)
    # cst5 = np.array([data_dom5[1:cst_order+2],data_dom5[cst_order+2:(cst_order+1)*2+1]])
    # # psi5_rp = (coords34_up[:, 0] - data_dom5[-5])/(data_dom5[-4] - data_dom5[-5])
    # psi5_rp = np.linspace(0, 1, j_dom34)
    # coords5 = reconstruct(psi5_rp, cst5, cst_order, data_dom5[-5], data_dom5[-4], data_dom5[-3], data_dom5[-2], data_dom5[-1], N1, N2, j_dom34)
    # coords5_up = coords5[:j_dom34, :][::-1, :]
    # coords5_up_ry = np.ones_like(coords5_up[:, 0])*y_end_dom5
    # coords5_rup = np.concatenate((coords5_up[:,0].reshape([-1,1]), coords5_up_ry.reshape([-1,1]), coords5_up[:,1].reshape([-1,1])), axis=1)
    # dom_5 = np.zeros([2, j_dom34, 3])
    # dom_5[0, :] = dom_3[-1, :]
    # dom_5[1, :] = coords5_rup
    # dom_5[1, 0] = dom_1[-1, -1]

    # coords5_low = coords5[j_dom34:, :]
    # coords5_low_ry = np.ones_like(coords5_low[:, 0])*y_end_dom5
    # dom_6 = np.zeros([2, j_dom34, 3])
    # dom_6[0, :] = dom_4[-1, :]
    # dom_6[1, :] = coords5_rlow
    # dom_6[1, 0] = dom_2[-1, -1]

    dom_6 = np.zeros([i_dom67, j_dom67, 3])
    dom_7 = np.zeros([i_dom67, j_dom67, 3])
    y_begin_dom6 = dom_1[-1, -1, 1]
    y_end_dom6 = mesh_para[-1, 0]
    y_pingjun = np.linspace(y_begin_dom6, y_end_dom6, i_dom67)
    qianyuan_x = si.interp1d(interpolated_cst[:, 0], interpolated_cst[:, -5], kind=2)(y_pingjun)
    qianyuan_z = si.interp1d(interpolated_cst[:, 0], interpolated_cst[:, -3], kind=2)(y_pingjun)
    qianyuan_y = y_pingjun
    qianyuan = np.concatenate((qianyuan_x.reshape([-1,1]), qianyuan_y.reshape([-1,1]), qianyuan_z.reshape([-1,1])), axis=1)
    for i in range(i_dom67):
        para = interpolate_cst_single(mesh_para, y_pingjun[i])
        cst_up = para[1:cst_order+2]
        cst_low = para[cst_order+2:(cst_order+1)*2+1]
        x1 = np.linspace(para[-5], para[-4], j_dom67)
        x01 = (np.linspace(para[-5], para[-4], j_dom67) - para[-5])/(para[-4] - para[-5])
        z_up =[]
        z_low =[]
        for j in range(j_dom67):
            z_tmp = cst_single(x01[j], cst_up, cst_order, para[-2])*(para[-4] - para[-5]) + para[-3]
            z_up.append(z_tmp)
            z_tmp = cst_single(x01[j], cst_low, cst_order, para[-2])*(para[-4] - para[-5]) + para[-3]
            z_low.append(z_tmp)
        z_up = np.array(z_up)
        z_low = np.array(z_low)
        y = np.ones_like(x01)*y_pingjun[i]
        dom_6[i] = np.concatenate((x1.reshape([-1,1]), y.reshape([-1,1]), z_up.reshape([-1,1])), axis=1)
        dom_7[i] = np.concatenate((x1.reshape([-1,1]), y.reshape([-1,1]), z_low.reshape([-1,1])), axis=1)
    dom_6[:, 0] = qianyuan
    dom_7[:, 0] = qianyuan
    
    dom_6[0] = dom_3[-1]
    # dom_7 = np.append(dom_4[-1].reshape([1, -1, 3]), dom_7, axis=0)
    dom_7[0, :] = dom_4[-1, :]

    dom_5 = dom_6[:2]
    dom_6 = dom_6[1:]


    dom_8 = np.zeros([2, j_dom67, 3])
    dom_8[0] = dom_6[-1]
    dom_8[1] = dom_7[-1]

    
    dom_9 = np.zeros([i_dom910, j_dom910, 3])
    dom_10 = np.zeros([i_dom910, j_dom910, 3])
    x910_begin = dom_3[0, -1, 0]
    x910_end = data0[-4]
    x_pingjun = np.linspace(x910_begin, x910_end, j_dom910)
    psi910_pingjun = (x_pingjun - data0[-5])/(data0[-4] - data0[-5])
    print(psi910_pingjun)

    coords910 = reconstruct(psi910_pingjun, cst, cst_order, data0[-5], data0[-4], data0[-3], data0[-2], data0[-1], N1, N2, j_dom910)
    coords910_pingjun = coords910[:j_dom910, :][::-1, :]
    coords910_pingjun_low = coords910[j_dom910:, :]
    coords910_y = np.zeros_like(coords910_pingjun[:,0])
    coords910_up = np.concatenate((coords910_pingjun[:,0].reshape([-1,1]), coords910_y.reshape([-1,1]), coords910_pingjun[:,1].reshape([-1,1])), axis=1)
    coords910_low = np.concatenate((coords910_pingjun_low[:,0].reshape([-1,1]), coords910_y.reshape([-1,1]), coords910_pingjun_low[:,1].reshape([-1,1])), axis=1)
    """"""
    houyuan_y = interpolated_cst[:, 0]
    
    for idx in range(len(houyuan_y)):
        if houyuan_y[idx] > dom_3[-1, -1, 1]:
            break
    houyuan_x = interpolated_cst[:, -4][:idx+1][::-1]
    houyuan_y = houyuan_y[:idx+1][::-1]
    houyuan_z = (interpolated_cst[:, -3] + interpolated_cst[:, -1]*(interpolated_cst[:, -4] - interpolated_cst[:, -5]))[:idx+1][::-1]

    print("houyuan_x:", houyuan_x[0], houyuan_x[-1], "x_pingjun:", x_pingjun[0], x_pingjun[-1])
    x_pingjunt = x_pingjun[1:]
    houyuan_y = si.interp1d(houyuan_x, houyuan_y, kind=1)(x_pingjunt)
    houyuan_z = si.interp1d(houyuan_x, houyuan_z, kind=1)(x_pingjunt)
    houyuan_x = x_pingjunt
    houyuan = np.concatenate((houyuan_x.reshape([-1,1]), houyuan_y.reshape([-1,1]), houyuan_z.reshape([-1,1])), axis=1)
    houyuan = np.append(dom_3[-1, -1].reshape([1,3]), houyuan, axis=0)
    """"""
    for j in range(j_dom910):
        y_list = np.array([0])
        z_up_list = np.array([coords910_up[j, 2]])
        z_low_list = np.array([coords910_low[j, 2]])
        # for da in interpolated_cst[1:][:idx]:
        #     if da[-4] > x_pingjun[j]:
        #         y_list = np.append(y_list, da[0])
        #         cst_up = da[1:cst_order+2]
        #         cst_low = da[cst_order+2:(cst_order+1)*2+1]
        #         chord_da = da[-4] - da[-5]
        #         x01 = (coords910_up[j, 0] - da[-5])/chord_da
        #         z_up = cst_single(x01, cst_up, cst_order, da[-2])*chord_da + da[-3]
        #         z_low = cst_single(x01, cst_low, cst_order, da[-1])*chord_da + da[-3]
        #         z_up_list = np.append(z_up_list, z_up)
        #         z_low_list = np.append(z_low_list, z_low)
        y_list = np.append(y_list, houyuan[j, 1])
        z_up_list = np.append(z_up_list, houyuan[j, 2])
        z_low_list = np.append(z_low_list, houyuan[j, 2])
        if y_list[0] == y_list[1]:
            y_up_new = np.ones(i_dom910)*y_list[0]
            y_low_new = np.ones(i_dom910)*y_list[0]
            z_up_new = np.ones(i_dom910)*z_up_list[0]
            z_low_new = np.ones(i_dom910)*z_low_list[0]
        else:
            print(y_list)
            y_up_new, z_up_new = redistribution(y_list, z_up_list, i_dom910)
            y_low_new, z_low_new = redistribution(y_list, z_low_list, i_dom910)

        for i in range(i_dom910):
            dom_9[i, j] = np.array([x_pingjun[j], y_up_new[i], z_up_new[i]])
            dom_10[i, j] = np.array([x_pingjun[j], y_low_new[i], z_low_new[i]])

        ##网格封闭性处理
        dom_9[:, 0] = dom_3[:, -1]
        dom_10[:, 0] = dom_4[:, -1]

    # # #钝底网格
    # # dom_11 = np.zeros([2, i_dom910, 3])
    # # dom_11[0, :] = dom_9[:, -1]
    # # dom_11[1, :] = dom_10[:, -1]

    ##尾涡面网格
    dom_12 = np.zeros([2, j_dom910+1, 3])
    dom_12[0, :-1] = dom_9[-1, :]
    dom_12[0, -1] = dom_9[-1, -1]
    dom_12[1, :] = dom_5[-1, -1]
    dom_12[1, :-1, 0] = dom_9[-1, :, 0]
    dom_12[:, 0] = dom_5[:, -1]
    dom_12[:, -1, 0] = 100

    ##网格方向修正
    dom_1 = dom_1.swapaxes(0, 1)
    dom_2 = dom_2[::-1, :].swapaxes(0, 1)
    dom_3 = dom_3.swapaxes(0, 1)
    dom_4 = dom_4[::-1, :].swapaxes(0, 1)
    dom_5 = dom_5.swapaxes(0, 1)
    dom_6 = dom_6.swapaxes(0, 1)
    dom_7 = dom_7[::-1, :].swapaxes(0, 1)
    dom_8 = dom_8.swapaxes(0, 1)
    dom_9 = dom_9.swapaxes(0, 1)
    dom_10 = dom_10[::-1, :].swapaxes(0, 1)
    # dom_11 = dom_11
    dom_12 = dom_12.swapaxes(0, 1)


    #mesh生成
    mesh_test([dom_1, dom_2, dom_3, dom_4, dom_5, dom_6, dom_7, dom_8, dom_9, dom_10, dom_12], r'FABOOM_test\indata\geo.x')

    print("网格生成完毕")

def quyitiaoxian(mesh_para, x):
    mesh_para = interpolate_cst3(mesh_para, 41)
    y_list = []
    z_list = []
    for da in mesh_para:
        if da[-5] <= x <= da[-4]:
            y_list.append(da[0])
            cst_up = da[1:14]
            chord_da = da[-4] - da[-5]
            x01 = (x - da[-5])/(chord_da)
            z_tmp = cst_single(x01, cst_up, 12, da[-2])*chord_da + da[-3]
            z_list.append(z_tmp)
    
    y_list = -1*np.array(y_list[::-1])
    y_list = np.append(y_list, -1*np.array(y_list[::-1]))
    z_list = z_list[::-1]
    z_list.extend(z_list[::-1])
    plt.plot(y_list, z_list, marker='o')
    plt.show()

if __name__ == "__main__":
    mesh_para = pd.read_csv(r"mesh_para\increase_cabin.csv").to_numpy()
    mesh_fenkuai(mesh_para)
    # quyitiaoxian(mesh_para, 50)
    # plt.show()