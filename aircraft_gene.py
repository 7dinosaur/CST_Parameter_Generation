import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import ndarray
import scipy.interpolate as si
from scipy.special import comb
from matplotlib import pyplot as plt
from cal_Lift import cal_Lift, cal_PLdB
import time

def deri_1d(x, y):
    assert len(x) == len(y), "xy应有相同形状"
    deri = np.zeros([len(x),])
    for i in range(len(x)):
        if i == 0:
            deri[i] = (-3*y[i] + 4*y[i+1] - y[i+2])/(x[i+2] - x[i])
        elif i == len(x)-1:
            deri[i] = (3*y[i] - 4*y[i-1] + y[i-2])/(x[i] - x[i-2])
        else:
            deri[i] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])

    return deri

def deri2_1d(x, y):
    assert len(x) == len(y), "xy应有相同形状"
    n = len(x)
    deri2 = np.zeros([n,])
    
    for i in range(n):
        if i == 0:
            dx = (x[i+3] - x[i])/3
            deri2[i] = (2*y[i] - 5*y[i+1] + 4*y[i+2] - y[i+3]) / (dx**2)
        elif i == n-1:
            dx = (x[i] - x[i-3])/3
            deri2[i] = (2*y[i] - 5*y[i-1] + 4*y[i-2] - y[i-3]) / (dx**2)
        else:
            dx = 0.5*(x[i+1] - x[i-1])
            deri2[i] = (y[i+1] - 2*y[i] + y[i-1]) / (dx**2)
    
    return deri2

class Aircraft:
    def __init__(self, origin_para = np.zeros([2, 2])) -> None:
        self.origin_para:ndarray = origin_para
        self.cst_order:int = int(0.5 * (self.origin_para.shape[1] - 8))
        self.N1 = 0.5
        self.N2 = 1
        self.air_mesh:NDArray = np.array([0])
        self.panel_mesh:NDArray = np.array([0])

    def read_from_csv(self, csv_file : str) -> None: #从csv文件读取参数
        mesh_para = pd.read_csv(csv_file).to_numpy()
        self.origin_para:ndarray = mesh_para
        self.cst_order = int(0.5 * (self.origin_para.shape[1] - 8))
    
    def interp_para(self, num_span, ori_para = None) -> ndarray: #补全对称条件并插值参数列表
        if ori_para is None:
            ori_para = self.origin_para
        y_list = ori_para[:, 0]
        full_y_list = np.append(-y_list[1:][::-1], y_list)
        full_y = np.linspace(y_list[0], y_list[-1], num_span)
        full_para = full_y.reshape([-1,1])

        for j in range(self.origin_para.shape[1]-1):
            full_mesh_para = np.append(ori_para[:,j+1][1:][::-1], ori_para[:,j+1])
            kind = 'linear' if j == ori_para.shape[1]-5 else 'quadratic'
            f = si.interp1d(full_y_list, full_mesh_para, kind=kind)
            para = f(full_y).reshape([-1,1])
            full_para = np.append(full_para, para, axis=1)
        # self.interped_para = full_para
        return full_para
    
    def interp_single_para(self, y) -> ndarray: #插值单一剖面参数
        ori_para = self.origin_para
        y_list = ori_para[:, 0]
        full_y_list = np.append(-y_list[1:][::-1], y_list)
        full_para = np.empty([self.origin_para.shape[1],])
        full_para[0] = y

        for j in range(self.origin_para.shape[1]-1):
            full_mesh_para = np.append(ori_para[:,j+1][1:][::-1], ori_para[:,j+1])
            kind = 'linear' if j == ori_para.shape[1]-5 else 'quadratic' ## 后缘保持一阶
            f = si.interp1d(full_y_list, full_mesh_para, kind=kind)
            para = f(y)
            full_para[j+1] = para

        return full_para
    
    def cst_rec(self, para, N1=0.5, N2=1, n_points=60, psi_end=1.0):
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

        coord_u = np.array([x_true,y_upper])
        coord_l = np.array([x_true, y_lower])

        return coord_u, coord_l

    def gene_simple_mesh(self, num_span, num_chord) -> ndarray:
        """生成三维网格数组,第一维为dom编号,如aircraft[0]=dom1,二三维为ij方向,四维[x,y,z]"""
        """simple_mesh只有两个dom,上下表面"""
        this_para = self.interp_para(num_span) ##网格展向尺度由插值后的参数列表长度决定
        mesh = np.zeros([2, num_span, num_chord, 3])
        order = self.cst_order

        for idx, data in enumerate(this_para):
            mesh[:, idx, :, 1] = data[0]
            coord_u, coord_l = self.cst_rec(data, self.N1, self.N2, num_chord)
            mesh[0, idx, :, [0, 2]] = coord_u
            mesh[1, idx, :, [0, 2]] = coord_l

        self.air_mesh = mesh

        return mesh
    
    def gene_panel_mesh(self, aoa:float = 3.0) -> list[ndarray]:
        """生成三维网格数组,第一维为dom编号,如aircraft[0]=dom1,二三维为ij方向,四维[x,y,z]"""
        """panel_mesh是可以直接输入faboom程序计算的分块网格"""
        def redistribution(x, y, n):
            xtmp = np.linspace(x[0], x[-1], 100)
            x = np.append(-np.array(x)[1:][::-1], np.array(x))
            y = np.append(np.array(y)[1:][::-1], np.array(y))

            fx = si.Akima1DInterpolator(x, y)

            y_tmp = fx(xtmp)
            L = np.zeros(100)
            for i in range(100 - 1):
                dx = xtmp[i+1] - xtmp[i]
                dy = y_tmp[i+1] - y_tmp[i]
                L[i+1] = (dx**2 + dy**2)**0.5 + L[i]
            l_total = L[-1]
            L_inv = si.interp1d(L, xtmp, kind='quadratic')
            dL = l_total/(n-1)
            L_pingjun = []
            for i in range(n):
                L_pingjun.append(dL*i)
            x_pingjun = np.append(L_inv(L_pingjun[:-1]), x[-1])
            y_pingjun = fx(x_pingjun)

            return x_pingjun, y_pingjun

        this_para = self.interp_para(61) ##此处插值仅保证曲线光滑，实际网格尺度与插值长度无关
        order = self.cst_order

        ##统一网格尺度设置
        nose_i, body_i, tail_i = 31, 60, 10
        nose_j = body_j = tail_j = 10
        wing_i = body_i
        wing_j = 29

        ##头部网格计算
        dom1, dom2 = np.zeros([nose_i, nose_j, 3]), np.zeros([nose_i, nose_j, 3]) ##机头网格
        dom3, dom4 = np.zeros([body_i, body_j, 3]), np.zeros([body_i, body_j, 3]) ##机身网格
        dom5 = np.zeros([body_i, 2, 3]) ##机翼上表面与尾涡面相连的网格
        dom6, dom7 = np.zeros([wing_i, wing_j-1, 3]), np.zeros([wing_i, wing_j, 3]) ##机翼网格
        dom8 = np.zeros([body_i, 2, 3]) ##翼尖网格
        dom9, dom10 = np.zeros([tail_i, tail_j, 3]), np.zeros([tail_i, tail_j, 3]) ##尾部网格
        dom11 = np.zeros([tail_j, 2, 3]) ##钝底网格
        dom12 = np.zeros([tail_i+1, 2, 3]) ##尾涡面网格

        ##机头网格计算
        #===================================#
        leading_edge_x = this_para[:, 2*order+3]
        leading_edge_y = this_para[:, 0]
        leading_edge_z = this_para[:, 2*order+5]
        f_leading_xy = si.interp1d(leading_edge_x, this_para[:, 0], kind='quadratic')
        f_leading_xz = si.interp1d(leading_edge_x, leading_edge_z, kind='quadratic')
        leading_deri = deri_1d(leading_edge_x, this_para[:, 0])
        mask = (leading_edge_x > 3)&(leading_deri > 0.12)&(leading_edge_y > 3.2)
        idx = np.argmax(mask)
        dom1_end = leading_edge_x[idx] ##自动选择网格切分点
        dom1_start = leading_edge_x[0]

        x_list = np.linspace(dom1_start, dom1_end, nose_i)
        delta_y = this_para[1, 0] - this_para[0, 0]
        dom1[0, :, 0] = dom2[0, :, 0] = this_para[0, -5]
        dom1[0, :, 1] = dom2[0, :, 1] = this_para[0, 0]
        dom1[0, :, 2] = dom2[0, :, 2] = this_para[0, -3]
        for i, x in enumerate(x_list[1:]):
            this_y_end = f_leading_xy(x)
            this_z_end = f_leading_xz(x)
            mask = this_para[:, 0] < this_y_end - 0.1*delta_y
            tmp_para = this_para[mask].copy() #获得从对称面到结束位置的参数
            coords_this = np.zeros([tmp_para.shape[0]+1, 4])
            coords_this[:, 0] = x
            coords_this[:, 1] = np.append(tmp_para[:, 0], this_y_end)
            for idx, da in enumerate(tmp_para):
                psi_end = (x - tmp_para[idx, -5])/(tmp_para[idx, -4] - tmp_para[idx, -5])
                z_u, z_l = self.cst_rec(da, self.N1, self.N2, 2, psi_end)
                coords_this[idx, 2] = z_u[1, -1]
                coords_this[idx, 3] = z_l[1, -1]
            coords_this[-1, 2] = this_z_end
            coords_this[-1, 3] = this_z_end
            new_coords = np.ones([nose_j, 3]) * x
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 2], nose_j)
            dom1[i+1] = new_coords
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 3], nose_j)
            dom2[i+1] = new_coords
        dom2 = dom2[:, ::-1]
        #===================================#

        ##机身网格计算
        #===================================#
        ##从后缘曲线截取与机头结束y值相等的x值，确定x范围
        x_begin = dom1_end
        trailing_edge_x = this_para[:, -4]
        f_trailing_yx = si.interp1d(this_para[:, 0], trailing_edge_x, kind='quadratic')
        x_end = f_trailing_yx(this_y_end)
        x_list = np.linspace(x_begin, x_end, body_i)
        wing_line = self.interp_single_para(this_y_end) ##机翼网格边界可复用
        end_u, end_l = self.cst_rec(wing_line, self.N1, self.N2, len(x_list))
        for i, x in enumerate(x_list):
            mask = this_para[:, 0] < this_y_end - 0.1*delta_y
            tmp_para = this_para[mask].copy() #获得从对称面到结束位置的参数
            coords_this = np.zeros([tmp_para.shape[0]+1, 4])
            coords_this[:, 0] = x
            coords_this[:, 1] = np.append(tmp_para[:, 0], this_y_end)
            for idx, da in enumerate(tmp_para):
                psi_end = (x - tmp_para[idx, -5])/(tmp_para[idx, -4] - tmp_para[idx, -5])
                z_u, z_l = self.cst_rec(da, self.N1, self.N2, 2, psi_end)
                coords_this[idx, 2] = z_u[1, -1]
                coords_this[idx, 3] = z_l[1, -1]
            coords_this[-1, 2] = end_u[1, i]
            coords_this[-1, 3] = end_l[1, i]
            new_coords = np.ones([body_j, 3]) * x
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 2], body_j)
            dom3[i] = new_coords
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 3], body_j)
            dom4[i] = new_coords
        dom4 = dom4[:, ::-1]
        #===================================#

        ##机翼网格计算
        #===================================#
        mask = this_para[:, 0] > this_y_end + 0.1*delta_y
        tmp_para = this_para[mask].copy() #获得从翼身交界面到翼尖位置的参数列表
        tmp_para = np.append(wing_line.reshape([1, -1]), tmp_para, axis=0)
        tmp_para = self.interp_para(wing_j, tmp_para)
        for j, pa in enumerate(tmp_para):
            y = pa[0]
            wing_u, wing_l = self.cst_rec(pa, self.N1, self.N2, wing_i)
            if j <= 1:
                dom5[:, j, 1] = y
                dom5[:, j, [0, 2]] = wing_u.T
            if j >= 1:
                dom6[:, j-1, 1] = y
                dom6[:, j-1, [0, 2]] = wing_u.T
            if j == len(tmp_para) - 1:
                dom8[:, :, 1] = y
                dom8[:, 0, [0, 2]] = wing_u.T
                dom8[:, 1, [0, 2]] = wing_l.T
            dom7[:, j, 1] = y
            dom7[:, j, [0, 2]] = wing_l.T
        dom7 = dom7[:, ::-1]
        #===================================#

        ##尾部网格计算
        #===================================#
        x_begin = x_end
        x_end = this_para[0, -4]
        x_list = np.linspace(x_begin, x_end, tail_i)
        end_point, _ = self.cst_rec(this_para[0, :], self.N1, self.N2, 2)
        end_point = np.array([x_end, this_para[0, 0], end_point[-1, 1]])
        for j in range(tail_j):
            begin_point_up = dom3[-1, j]
            begin_point_low = dom4[-1, j]
            for i, x in enumerate(x_list):
                x_psi = (x - x_begin)/(x_end - x_begin)
                dom9[i, j, 0] = dom10[i, j, 0] = x
                dom9[i, j, 1:] = begin_point_up[1:] + x_psi*(end_point[1:] - begin_point_up[1:])
                dom10[i, j, 1:] = begin_point_low[1:] + x_psi*(end_point[1:] - begin_point_low[1:])
        #===================================#

        ##钝底网格...呃...赋值
        #===================================#
        # dom11[:, 0] = dom9[-1]
        # dom11[:, 1] = dom10[-1, ::-1]
        # dom11 = dom11.transpose(1, 0, 2)
        #===================================#

        ##尾涡面网格计算
        #===================================#
        dom12[:-1, 0] = dom9[:, -1]##贴上尾部网格边缘
        dom12[:, 1, [1, 2]] = dom5[-1, 1, [1, 2]]
        dom12[:-1, 1, 0] = dom12[:-1, 0, 0]; dom12[-1, 1, 0] = 100
        dom12[0, 1] = dom5[-1, 1]
        dom12[-1, 0, [1, 2]] = dom9[-1, 1, [1, 2]]
        dom12[-1, :, 0] = 100
        #===================================#

        ## 旋转网格
        theta = np.radians(-aoa)
        c, s = np.cos(theta), np.sin(theta)
        for dom_name in [f"dom{i}" for i in range(1, 13)]:
            if dom_name in locals():
                mesh = locals()[dom_name]
                x, z = mesh[..., 0], mesh[..., 2]
                mesh[..., 0], mesh[..., 2] = x*c - z*s, x*s + z*c
        print(f"攻角为{aoa}")
        
        panel_mesh = [locals()[f"dom{i}"] for i in range(1, 13) if i != 11] ##由于分块网格长度尺度不统一用列表存储

        return panel_mesh
    
    def write_mesh(self, mesh_type:str = "panel", file_path:str = "geo.x", aoa = 3.0) -> None:
        """自动识别网格类型并写入文件"""
        with open(file_path, 'w') as f:
            if mesh_type == "simple": #simple mesh
                mesh = self.gene_simple_mesh(41, 60)
                print("写入一般网格...")
                n_dom = mesh.shape[0]
                n_i = mesh.shape[1]
                n_j = mesh.shape[2]
                f.write(f"{n_dom}\n")
                f.write(f"{n_i} {n_j} 1\n")
                f.write(f"{n_i} {n_j} 1\n")
                for dom in mesh:
                    dom = dom.transpose(2, 1, 0).flatten().reshape([-1, 5])
                    for line in dom:
                        f.write(" ".join(f"{x:.6f}" for x in line) + "\n")
                print(f"写入完毕,网格形状为[{n_i},{n_j}]. 网格文件路径：{file_path}")

            elif mesh_type == "panel": #panel mesh
                mesh = self.gene_panel_mesh(aoa)
                print("写入面元网格...")
                f.write(f"{len(mesh)}\n")
                for dom in mesh:
                    n_i, n_j = dom.shape[1], dom.shape[0]
                    f.write(f"{n_i} {n_j} 1\n")
                for dom in mesh:
                    dom = dom.transpose(2, 0, 1)
                    row_size = 4  # 控制每行4个元素
                    for coord in dom:
                        coord = coord.flatten()
                        for i in range(0, len(coord), row_size):
                            line_elements = coord[i:i+row_size]
                            line_str = " ".join(f"{x:.6f}" for x in line_elements)
                            f.write(line_str + "\n")
                print(f"写入完毕,面元数为[{len(mesh)}]. 网格文件路径：{file_path}")

    def cal_volume(self):
        """从对称面开始扫描索引,基准为2座位,每次增加一个座位,并评估载客量最多的方案"""
        height_cabin = 2.0 #客舱高度为2m
        seats_width = 0.5 + 0.1 #座椅宽度为0.5m,间距0.1m
        seats_length = 0.97 #座椅前后长度
        aisle_width = 0.5 #过道宽度为0.5m
        scan_range = range(2, 7) #扫描范围从2个座位到6个座位
        prec = 100
        passengers = []
        z_step = 0.2 #垂向扫描步长
        
        for n_seats in scan_range:
            cabin_width = aisle_width + n_seats * seats_width
            half_width = 0.5 * cabin_width
            sym_para = self.origin_para[0]; end_para = self.interp_single_para(half_width)
            cu_end, cl_end = self.cst_rec(end_para, self.N1, self.N2, prec)
            len_sym = []
            cu = cu_end.copy().T
            for j in range(prec):
                cu[j, 1] = cu_end[1, j] - cl_end[1, j]
                if cu[j, 1] > height_cabin:
                    n_scan = int((cu[j, 1] - height_cabin)/z_step) + 1
                    k_list = []
                    for r in range(n_scan):
                        x_start = cu[j, 0]
                        psi_sym = (x_start - sym_para[-5])/(sym_para[-4] - sym_para[-5])
                        zu_sym, zl_sym = self.cst_rec(sym_para, self.N1, self.N2, 2, psi_end=psi_sym)
                        z_u = cu_end[1, j] - z_step*r
                        z_l = z_u - height_cabin
                        if zu_sym[-1, 1] < z_u or zl_sym[-1, 1] > z_l:
                            break
                        for k in range(j+1, prec):
                            x_end = cu[k, 0]
                            psi_sym = (x_end - sym_para[-5])/(sym_para[-4] - sym_para[-5])
                            zu_sym, zl_sym = self.cst_rec(sym_para, self.N1, self.N2, 2, psi_end=psi_sym)
                            if (cu_end[1, k] < z_u or cl_end[1, k] > z_l) or (zu_sym[-1, 1] < z_u or zl_sym[-1, 1] > z_l):
                                k_list.append(cu[k, 0])
                                break
                    if len(k_list) > 0:
                        x_end = max(k_list)
                        len_sym.append([x_start, x_end, x_end-x_start])
            if len(len_sym) > 0:
                len_sym = np.array(len_sym)
                max_len = np.argmax(len_sym[:, 2])
                max_len = len_sym[max_len][2]
                n_row = int(max_len/seats_length)
                # print(max_row)
                # print(f"每排{n_seats}座: 客舱宽度为{cabin_width}, 长度为{max_len:.2f}, 容纳排数{n_row}, 载客量{n_row*n_seats}")
                passengers.append(n_row*n_seats)
            else:
                passengers.append(0)

        return max(passengers)
    
    def if_smooth(self, fig = False):
        mesh = self.gene_simple_mesh(41, 31)
        if fig:
            plt.figure(figsize=(16, 5))
        n_turn_u = 0
        n_turn_l = 0
        for i in range(1, 3):
            line_u = mesh[0, :, i*10]
            line_l = mesh[1, :, i*10]
            deri_u = np.gradient(line_u[:, 2], line_u[:, 1])
            deri_l = np.gradient(line_l[:, 2], line_l[:, 1])
            deri2_u = np.gradient(deri_u, line_u[:, 1])
            deri2_l = np.gradient(deri_l, line_l[:, 1])
            for i in range(len(deri2_u)-1):
                if (deri2_u[i] * deri2_u[i+1] < 0):
                    n_turn_u += 1
                if (deri2_l[i] * deri2_l[i+1] < 0):
                    n_turn_l += 1
            if fig:
                plt.subplot(1, 3, 1)
                plt.plot(line_u[:, 1], line_u[:, 2])
                plt.plot(line_l[:, 1], line_l[:, 2])
                plt.subplot(1, 3, 2)
                plt.plot(line_u[:, 1], deri_u)
                plt.plot(line_l[:, 1], deri_l)
                plt.subplot(1, 3, 3)
                plt.plot(line_u[:, 1], deri2_u)
                plt.plot(line_l[:, 1], deri2_l)
        
        return n_turn_u, n_turn_l

    def Laplace(self) -> list[float]:
        span, chord = 61, 61
        mesh_u, mesh_l = self.gene_simple_mesh(span, chord)

        laplace_norm_u = self.Laplace_single(mesh_u)
        laplace_norm_l = self.Laplace_single(mesh_l)

        return [laplace_norm_u, laplace_norm_l]
    
    def Laplace_panel(self) -> list[float]:
        panel_mesh = self.gene_panel_mesh(0)
        laplace_norms = []
        for dom in panel_mesh:
            laplace_norms.append(self.Laplace_single(dom))

        return laplace_norms
    
    def Laplace_single(self, mesh) -> float:

        laplace = np.zeros_like(mesh)

        # ===== 中间区域 (有上下左右四个邻居) =====
        center = (
            mesh[:-2, 1:-1] + mesh[2:, 1:-1] +
            mesh[1:-1, :-2] + mesh[1:-1, 2:]
        ) / 4.0

        laplace[1:-1, 1:-1] = mesh[1:-1, 1:-1] - center

        # ===== 边界处理 =====
        # 上下边（不含角）
        center = (
            mesh[[0, -1], 0:-2] + mesh[[0, -1], 2:]
        ) / 2.0
        laplace[[0, -1], 1:-1] = mesh[[0, -1], 1:-1] - center

        # # 左右边（不含角）
        center = (
            mesh[0:-2, [0, -1]] + mesh[2:, [0, -1]]
        ) / 2.0
        laplace[1:-1, [0, -1]] = mesh[1:-1, [0, -1]] - center

        # ===== 计算范数 =====
        laplace_norm = np.sum(laplace ** 2)

        return laplace_norm
    
    def Laplace_seperate(self) -> list[float]:
        span, chord = 121, 61
        mesh_u, mesh_l = self.gene_simple_mesh(span, chord)

        span_laplace_u = np.zeros_like(mesh_u)
        chord_laplace_u = np.zeros_like(mesh_u)
        span_laplace_l = np.zeros_like(mesh_l)
        chord_laplace_l = np.zeros_like(mesh_l)

        # ===== 中间区域 (有上下左右四个邻居) =====
        ##span方向laplace, 第一个维度代表span, 所以在第一个维度作差值, 不做差值的维度应该置1:-1
        center_u = (
            mesh_u[:-2, 1:-1] + mesh_u[2:, 1:-1]
        ) / 2.0

        center_l = (
            mesh_l[:-2, 1:-1] + mesh_l[2:, 1:-1]
        ) / 2.0

        span_laplace_u[1:-1, 1:-1] = mesh_u[1:-1, 1:-1] - center_u
        span_laplace_l[1:-1, 1:-1] = mesh_l[1:-1, 1:-1] - center_l

        ##chord方向laplace, 第一个维度代表span, 所以在第二个维度作差值
        center_u = (
            mesh_u[1:-1, :-2] + mesh_u[1:-1, 2:]
        ) / 2.0

        center_l = (
            mesh_l[1:-1, :-2] + mesh_l[1:-1, 2:]
        ) / 2.0

        chord_laplace_u[1:-1, 1:-1] = mesh_u[1:-1, 1:-1] - center_u
        chord_laplace_l[1:-1, 1:-1] = mesh_l[1:-1, 1:-1] - center_l

        span_laplace_u_norm = np.sum(span_laplace_u ** 2)
        chord_laplace_u_norm = np.sum(chord_laplace_u ** 2)
        span_laplace_l_norm = np.sum(span_laplace_l ** 2)
        chord_laplace_l_norm = np.sum(chord_laplace_l ** 2)

        return [span_laplace_u_norm, chord_laplace_u_norm, span_laplace_l_norm, chord_laplace_l_norm]

if __name__ == "__main__":
    air_para = Aircraft()
    air_para.read_from_csv("smooth_test.csv")
    # air_para.write_mesh("panel", r"FABOOM_test\\indata\\geo.x", 4.0)
    print(air_para.if_smooth())
    # cal_Lift()
    para = pd.read_csv("no_lift_samples.csv").to_numpy()[:, 2:]
    for pa in para[:10]:
        air_para = Aircraft(pa.reshape([-1, 24]))
        print(air_para.Laplace())
        print(air_para.Laplace_panel())
        print(air_para.if_smooth())
        air_para.write_mesh("panel", r"check.x")
        input("Press Enter to continue...")
    # air_para.write_mesh("panel", r"geo.x")
    # lift = cal_Lift()

    # plt.show()