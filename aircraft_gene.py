import calendar
import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import ndarray
import scipy.interpolate as si
from scipy.special import comb
from matplotlib import pyplot as plt
from cal_Lift import cal_Lift, cal_PLdB
import time
import os

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
    
    def interp_para(self, num_span, ori_para = None, sym = True) -> ndarray: #补全对称条件并插值参数列表
        if ori_para is None:
            ori_para = self.origin_para
        if sym:
            y_list = ori_para[:, 0]
            full_y_list = np.append(-y_list[1:][::-1], y_list)
            full_y = np.linspace(y_list[0], y_list[-1], num_span)
            full_para = full_y.reshape([-1,1])

            for j in range(self.origin_para.shape[1]-1):
                full_mesh_para = np.append(ori_para[:,j+1][1:][::-1], ori_para[:,j+1])
                kind = 1 if j == ori_para.shape[1]-5 else 2
                f = si.interp1d(full_y_list, full_mesh_para, kind=kind)
                para = f(full_y).reshape([-1,1])
                full_para = np.append(full_para, para, axis=1)
                # if j == ori_para.shape[1]-6 or j == ori_para.shape[1]-5:
                #     plt.plot(para, full_y)
                #     plt.plot(para, -full_y)
        else:
            y_list = ori_para[:, 0]
            full_y = np.linspace(y_list[0], y_list[-1], num_span)
            full_para = full_y.reshape([-1,1])
            for j in range(self.origin_para.shape[1]-1):
                kind = 1 if j == ori_para.shape[1]-5 else 2
                f = si.interp1d(y_list, ori_para[:,j+1], kind=kind)
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
            kind = 'linear' if j == ori_para.shape[1]-5 else 2 ## 后缘保持一阶
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
    
    def reassign_cst_order(self, new_cst_order: int, fit_points: int = 100) -> ndarray:
        """
        重分配CST阶数，拟合新的CST系数并生成新的参数列表
        
        参数：
            new_cst_order: 新的CST阶数（正整数）
            fit_points: 拟合用的弦向采样点数，默认100足够精确
        
        返回：
            new_origin_para: 新的完整参数列表数组，形状与原参数列表一致，仅CST系数更新
        """
        # 原始参数与几何参数
        old_para = self.origin_para
        new_para_list = []
        
        # 遍历每一个展向截面，逐截面拟合新CST系数
        for sec_para in old_para:
            # 1. 用原始CST参数生成精确的翼型坐标点（用于拟合）
            coord_u, coord_l = self.cst_rec(sec_para, self.N1, self.N2, n_points=fit_points)
            x_coords, y_u_coords = coord_u
            x_coords, y_l_coords = coord_l
            
            # 提取原始固定几何参数（前缘、后缘、偏移、后缘偏移等）
            le = sec_para[-5]
            te = sec_para[-4]
            z_offset = sec_para[-3]
            dy_upper = sec_para[-2]
            dy_lower = sec_para[-1]
            chord = te - le
            psi = (x_coords - le) / chord  # 归一化弦长坐标
            
            # 2. 构建新阶数的Bernstein基函数矩阵
            n = new_cst_order
            B_new = np.zeros((fit_points, n + 1))
            for i in range(n + 1):
                B_new[:, i] = comb(n, i) * (psi ** i) * ((1 - psi) ** (n - i))
            # CST形状函数
            shape_func = (psi ** self.N1) * ((1 - psi) ** self.N2)
            B_shape = shape_func.reshape(-1, 1) * B_new  # 带形状函数的基矩阵
            
            # 3. 最小二乘拟合新的CST系数
            # 上表面拟合
            y_u_target = (y_u_coords - z_offset - psi * dy_upper) / chord
            new_coeffs_u = np.linalg.lstsq(B_shape, y_u_target, rcond=None)[0]
            # 下表面拟合
            y_l_target = (y_l_coords - z_offset - psi * dy_lower) / chord
            new_coeffs_l = np.linalg.lstsq(B_shape, y_l_target, rcond=None)[0]
            
            # 4. 拼接新的截面参数
            y_span = sec_para[0]  # 展向坐标
            new_sec_para = np.hstack([y_span, new_coeffs_u, new_coeffs_l,
                                     le, te, z_offset, dy_upper, dy_lower])
            new_para_list.append(new_sec_para)
        
        # 5. 生成新参数数组，更新类内属性
        new_origin_para = np.array(new_para_list)
        self.origin_para = new_origin_para
        self.cst_order = new_cst_order
        
        return new_origin_para

    def gene_simple_mesh(self, num_span=81, num_chord=120, aoa=0.0) -> ndarray:
        """生成三维网格数组,第一维为dom编号,如aircraft[0]=dom1,二三维为ij方向,四维[x,y,z]"""
        """simple_mesh只有两个dom,上下表面"""
        this_para = self.interp_para(num_span) ##网格展向尺度由插值后的参数列表长度决定
        mesh = np.zeros([2, num_span, num_chord, 3])

        for idx, data in enumerate(this_para):
            mesh[:, idx, :, 1] = data[0]
            coord_u, coord_l = self.cst_rec(data, self.N1, self.N2, num_chord)
            mesh[0, idx, :, [0, 2]] = coord_u
            mesh[1, idx, :, [0, 2]] = coord_l

        ## 旋转网格
        theta = np.radians(-aoa)
        c, s = np.cos(theta), np.sin(theta)
        x, z = mesh[..., 0], mesh[..., 2]
        mesh[..., 0], mesh[..., 2] = x*c - z*s, x*s + z*c
        # print(f"攻角为{aoa}")


        self.air_mesh = mesh

        return mesh
    
    def gene_panel_mesh(self, aoa:float = 3.0, shape = [31, 60, 10, 8, 27]) -> list[ndarray]:
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

        this_para = self.interp_para(62) ##此处插值仅保证曲线光滑，实际网格尺度与插值长度无关
        order = self.cst_order

        ##统一网格尺度设置
        nose_i, body_i, tail_i = shape[0], shape[1], shape[2]
        nose_j = body_j = tail_j = shape[3]
        wing_i = body_i
        wing_j = shape[4]

        ##头部网格计算
        dom1, dom2 = np.zeros([nose_i, nose_j, 3]), np.zeros([nose_i, nose_j, 3]) ##机头网格
        dom3, dom4 = np.zeros([body_i, body_j, 3]), np.zeros([body_i, body_j, 3]) ##机身网格
        dom5 = np.zeros([body_i, 2, 3]) ##机翼上表面与尾涡面相连的网格
        dom6, dom7 = np.zeros([wing_i, wing_j-1, 3]), np.zeros([wing_i, wing_j, 3]) ##机翼网格
        dom8 = np.zeros([body_i, 2, 3]) ##翼尖网格
        dom9, dom10 = np.zeros([tail_i, tail_j, 3]), np.zeros([tail_i, tail_j, 3]) ##尾部网格
        dom11 = np.zeros([tail_j, 2, 3]) ##钝底网格
        dom12 = np.zeros([tail_i, 2, 3]) ##尾涡面网格

        ##机头网格计算
        #===================================#
        leading_edge_x = this_para[:, 2*order+3]
        leading_edge_y = this_para[:, 0]
        leading_edge_z = this_para[:, 2*order+5]
        trailing_edge_x = this_para[:, -4]
        ## 以后缘曲率最大处（转折点）的y值为分块判据
        dte_dy = np.gradient(trailing_edge_x, leading_edge_y)
        d2te_dy2 = np.abs(np.gradient(dte_dy, leading_edge_y))
        idx_te = np.argmax(d2te_dy2[2:-2]) + 2  # 排除边界
        this_y_end = leading_edge_y[idx_te]
        f_y_lex = si.interp1d(leading_edge_y, leading_edge_x, kind='quadratic')
        dom1_end = f_y_lex(this_y_end)
        dom1_start = leading_edge_x[0]
        f_leading_xy = si.interp1d(leading_edge_x, leading_edge_y, kind='quadratic')
        f_leading_xz = si.interp1d(leading_edge_x, leading_edge_z, kind='quadratic')

        x_list = np.linspace(dom1_start, dom1_end, nose_i)
        delta_y = this_para[1, 0] - this_para[0, 0]
        dom1[0, :, 0] = dom2[0, :, 0] = this_para[0, -5]
        dom1[0, :, 1] = dom2[0, :, 1] = this_para[0, 0]
        dom1[0, :, 2] = dom2[0, :, 2] = this_para[0, -3]
        for i, x in enumerate(x_list[1:]):
            y_end_at_x = f_leading_xy(x)
            z_end_at_x = f_leading_xz(x)
            mask = this_para[:, 0] < y_end_at_x - 0.1*delta_y
            if not np.any(mask):
                mask[0] = True  # 至少保留对称面站位
            tmp_para = this_para[mask].copy()
            coords_this = np.zeros([tmp_para.shape[0]+1, 4])
            coords_this[:, 0] = x
            coords_this[:, 1] = np.append(tmp_para[:, 0], y_end_at_x)
            for idx, da in enumerate(tmp_para):
                psi_end = (x - tmp_para[idx, -5])/(tmp_para[idx, -4] - tmp_para[idx, -5])
                z_u, z_l = self.cst_rec(da, self.N1, self.N2, 2, psi_end)
                coords_this[idx, 2] = z_u[1, -1]
                coords_this[idx, 3] = z_l[1, -1]
            coords_this[-1, 2] = z_end_at_x
            coords_this[-1, 3] = z_end_at_x
            ## 上下表面共用上表面弧长分布的y坐标，避免交叉
            y_new, zu_new = redistribution(coords_this[:, 1], coords_this[:, 2], nose_j)
            y_orig = coords_this[:, 1]
            zl_orig = coords_this[:, 3]
            y_mir = np.append(-y_orig[1:][::-1], y_orig)
            zl_mir = np.append(zl_orig[1:][::-1], zl_orig)
            fx_l = si.Akima1DInterpolator(y_mir, zl_mir)
            zl_new = fx_l(y_new)
            new_coords_u = np.ones([nose_j, 3]) * x
            new_coords_u[:, 1] = y_new
            new_coords_u[:, 2] = zu_new
            dom1[i+1] = new_coords_u
            new_coords_l = np.ones([nose_j, 3]) * x
            new_coords_l[:, 1] = y_new
            new_coords_l[:, 2] = zl_new
            dom2[i+1] = new_coords_l
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
            if not np.any(mask):
                mask[0] = True  # 至少保留对称面站位
            tmp_para = this_para[mask].copy()
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
            ## 上下表面共用上表面弧长分布的y坐标，避免交叉
            y_new, zu_new = redistribution(coords_this[:, 1], coords_this[:, 2], body_j)
            y_orig = coords_this[:, 1]
            zl_orig = coords_this[:, 3]
            y_mir = np.append(-y_orig[1:][::-1], y_orig)
            zl_mir = np.append(zl_orig[1:][::-1], zl_orig)
            fx_l = si.Akima1DInterpolator(y_mir, zl_mir)
            zl_new = fx_l(y_new)
            new_coords_u = np.ones([body_j, 3]) * x
            new_coords_u[:, 1] = y_new
            new_coords_u[:, 2] = zu_new
            dom3[i] = new_coords_u
            new_coords_l = np.ones([body_j, 3]) * x
            new_coords_l[:, 1] = y_new
            new_coords_l[:, 2] = zl_new
            dom4[i] = new_coords_l
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
        end_u, end_l = self.cst_rec(this_para[0, :], self.N1, self.N2, 2)
        z_te = 0.5 * (end_u[1, -1] + end_l[1, -1])
        end_point = np.array([x_end, this_para[0, 0], z_te])
        for j in range(tail_j):
            begin_point_up = dom3[-1, j]
            begin_point_low = dom4[-1, j]
            for i, x in enumerate(x_list):
                x_psi = (x - x_begin)/(x_end - x_begin)
                dom9[i, j, 0] = dom10[i, j, 0] = x
                dom9[i, j, 1:] = begin_point_up[1:] + x_psi*(end_point[1:] - begin_point_up[1:])
                dom10[i, j, 1:] = begin_point_low[1:] + x_psi*(end_point[1:] - begin_point_low[1:])

        dom9 = dom9[:-1]
        dom10 = dom10[:-1]
        #===================================#

        ##钝底网格...呃...赋值
        #===================================#
        dom11[:, 1] = dom9[-1]
        dom11[:, 0] = dom10[-1, ::-1]
        dom11 = dom11[::-1, ::-1] ##翻转j方向使得dom11与dom9、dom10的连接关系正确
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
        # print(f"攻角为{aoa}")
        
        panel_mesh = [locals()[f"dom{i}"] for i in range(1, 13)] ##由于分块网格长度尺度不统一用列表存储

        return panel_mesh



    def write_mesh(self, mesh_type:str = "panel", file_path:str = "geo.x", aoa = 3.0) -> None:
        """自动识别网格类型并写入文件
        mesh_type: 'simple' | 'panel' | 'simple_nurbs' | 'panel_nurbs'
        """
        with open(file_path, 'w') as f:
            if mesh_type == "simple": #simple mesh
                mesh = self.gene_simple_mesh(81, 120)
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

            elif mesh_type == "simple_nurbs": #simple mesh (NURBS)
                mesh = self.gene_simple_mesh_nurbs(81, 120, aoa)
                print("写入NURBS一般网格...")
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

            elif mesh_type == "panel_nurbs": #panel mesh (NURBS)
                mesh = self.gene_panel_mesh_nurbs(aoa)
                print("写入NURBS面元网格...")
                f.write(f"{len(mesh)}\n")
                for dom in mesh:
                    n_i, n_j = dom.shape[1], dom.shape[0]
                    f.write(f"{n_i} {n_j} 1\n")
                for dom in mesh:
                    dom = dom.transpose(2, 0, 1)
                    row_size = 4
                    for coord in dom:
                        coord = coord.flatten()
                        for i in range(0, len(coord), row_size):
                            line_elements = coord[i:i+row_size]
                            line_str = " ".join(f"{x:.6f}" for x in line_elements)
                            f.write(line_str + "\n")
                print(f"写入完毕,面元数为[{len(mesh)}]. 网格文件路径：{file_path}")

    def gene_mesh_for_SU2(self, file_path:str = "geo.x", aoa = 3.0):
        mesh = self.gene_panel_mesh(aoa, shape=[41, 80, 20, 20, 41])[:-1]
        dom5 = np.concatenate([mesh[4], mesh[5][:, 1:]], axis=1)
        mesh[4:6] = [dom5]

        with open(file_path, 'w') as f:
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

    def export_airfoil_profiles(self, output_dir="airfoil_profiles", n_points=60):
        mesh_para = self.origin_para.copy()

        # 1. 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        exported_count = 0

        print(f"正在导出原始参数剖面的翼型文件到目录 '{output_dir}'...")
        
        # 2. 遍历mesh_para的每一行（每个原始剖面）
        for idx, data_line in enumerate(mesh_para):
            try:
                data_line[-5] = 0; data_line[-4] = 1; data_line[-3] = 0
                coords_u, coords_l = self.cst_rec(data_line, N1=0.5, N2=1, n_points=n_points)

                lower_surface_ordered = coords_l.T[::-1]   # 反转下表面顺序
                upper_surface_ordered = coords_u.T   # 反转上表面顺序
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

    def cal_volume(self):
        """从对称面开始扫描索引,基准为2座位,每次增加一个座位,并评估载客量最多的方案"""
        height_cabin = 2.0 #客舱高度为2m
        seats_width = 0.5 + 0.05 #座椅宽度为0.5m,间距0.1m
        seats_length = 0.97 #座椅前后长度
        aisle_width = 0.5 #过道宽度为0.5m
        scan_range = range(4, 6) #扫描范围从2个座位到6个座位
        prec = 100
        passengers = []
        z_step = 0.1 #垂向扫描步长
        
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
        mesh = self.gene_simple_mesh(81, 51)
        if fig:
            plt.figure(figsize=(16, 5))
        n_turn_u = 0.
        n_turn_l = 0.
        for i in range(6):
            line_u = mesh[0, :, i*10]
            line_l = mesh[1, :, i*10]
            deri_u = np.gradient(line_u[:, 2], line_u[:, 1])
            n_turn_u += (np.max(line_u[:, 2]) - np.min(line_u[:, 2]))
            deri_l = np.gradient(line_l[:, 2], line_l[:, 1])
            n_turn_l += (np.max(line_l[:, 2]) - np.min(line_l[:, 2]))
            deri2_u = np.gradient(deri_u, line_u[:, 1])
            n_turn_u += np.sum(deri2_u**2)
            deri2_l = np.gradient(deri_l, line_l[:, 1])
            n_turn_l += np.sum(deri2_l**2)
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
        
        return np.array([n_turn_u, n_turn_l])
    
    
    def tail_new(self):
        mesh = self.gene_panel_mesh(0)
        dom_tail_u = mesh[8]
        dom_tail_l = mesh[9]

        penalty = 0.0

        lu = dom_tail_u[0]
        ll = dom_tail_l[0][::-1]
        plt.plot(lu[:, 1], lu[:, 2])
        plt.plot(ll[:, 1], ll[:, 2])

        for j in range(8):
            if lu[j, 2] - ll[j, 2] < -1e-10:
                penalty += 1
                plt.scatter(lu[j, 1], lu[j, 2])
                plt.scatter(ll[j, 1], ll[j, 2])

        return penalty

    def Laplace(self) -> list[float]:
        mesh_u, mesh_l = self.air_mesh[0], self.air_mesh[1]

        laplace_norm_u = self.Laplace_single(mesh_u)
        laplace_norm_l = self.Laplace_single(mesh_l)

        return [laplace_norm_u, laplace_norm_l]
    
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

    def param_smoothness(self, normalize: bool = True, n_dense: int = 51) -> dict:
        """展向设计参数光顺度判据：弯曲能 + 拐点数

        直接对 origin_para 各列的展向分布做检查，而非对网格。
        先对每条参数曲线做三次样条插值到 n_dense 个点，再计算：
        - 弯曲能 ∫(d²p/dy²)² dy：惩罚曲率幅值大
        - 拐点数 d²p/dy² 符号变化次数：惩罚来回振荡（小幅高频也能捕获）

        两者互补 —— 小幅高频振荡弯曲能小但拐点多，大幅单弯拐点少但弯曲能大。
        插值到密网格保证拐点计数反映连续曲线行为，而非离散采样的数值噪声。
        normalize=True 时每个参数归一化到 [0,1] 后计算，使不同量纲可比。

        Returns:
            score:           float  综合得分（越低越光顺，可直接与 baseline 比较）
            bending_total:   float  总归一化弯曲能
            inflection_total: int   总拐点数
            details:         dict   逐参数明细 {name: (bending, n_infl)}
        """
        from scipy.interpolate import CubicSpline

        para = self.origin_para
        y = para[:, 0]
        order = self.cst_order
        y_dense = np.linspace(y[0], y[-1], n_dense)

        # 待检查的展向参数列
        param_cols = {}
        param_cols['le']       = para[:, -5]
        param_cols['te']       = para[:, -4]
        param_cols['chord']    = para[:, -4] - para[:, -5]
        param_cols['z_offset'] = para[:, -3]
        param_cols['dy_upper'] = para[:, -2]
        param_cols['dy_lower'] = para[:, -1]
        for i in range(order + 1):
            param_cols[f'cst_u_{i}'] = para[:, 1 + i]
            param_cols[f'cst_l_{i}'] = para[:, 1 + order + i]

        details = {}
        bending_total = 0.0
        inflection_total = 0

        for name, values in param_cols.items():
            # 三次样条插值到密网格，用样条解析二阶导（避免有限差分噪声）
            cs = CubicSpline(y, values)
            v_dense = cs(y_dense)
            d2 = cs(y_dense, 2)  # CubicSpline 解析二阶导

            v_range = np.max(v_dense) - np.min(v_dense)
            if normalize and v_range > 1e-10:
                v_dense = (v_dense - np.min(v_dense)) / v_range
                d2 = d2 / v_range  # 二阶导同步缩放

            # 弯曲能: ∫(d²p/dy²)² dy  梯形积分
            dy_dense = np.diff(y_dense)
            bend = float(np.sum(0.5 * (d2[:-1]**2 + d2[1:]**2) * dy_dense))

            # 拐点数: 二阶导数符号变化次数（忽略幅度 < 1e-8 的零穿越）
            signs = np.sign(d2)
            nonzero = signs[np.abs(d2) > 1e-8]
            n_infl = int(np.sum(np.abs(np.diff(nonzero)) > 0)) if len(nonzero) > 1 else 0

            details[name] = (bend, n_infl)

            # 几何参数权重 1.0，CST 系数权重 0.3
            is_geom = name in ('le', 'te', 'chord', 'z_offset', 'dy_upper', 'dy_lower')
            w = 1.0 if is_geom else 0.3
            bending_total += w * bend
            inflection_total += int(w * n_infl) if is_geom else (1 if n_infl > 1 else 0)

        # 拐点数的惩罚系数：1 个拐点 ≈ 一定量的弯曲能
        score = bending_total + inflection_total**2 * 0.5

        return {
            'score':           score,
            'bending_total':   bending_total,
            'inflection_total': inflection_total,
            'details':         details,
        }

    def surface_smoothness(self, n_chord_stations: int = 19) -> float:
        """基于等弦长点展向曲线的外形光顺度判据

        对 8 个剖面的 CST 曲线在 n_chord_stations 个等 psi 位置采样，
        每个弦向站上下表面各得一条展向 z 曲线（共 2*n_chord_stations 条），
        对每条曲线计算归一化弯曲能 + 拐点数后求和。

        仅返回一个 float，方便优化器中直接调用。
        若 19 站超 1s，改用 9 站。
        """
        from scipy.interpolate import CubicSpline

        para = self.origin_para
        n_sec = len(para)
        y_span = para[:, 0]

        # 1. 每个剖面生成 CST 曲线，只取 z 坐标
        z_up = np.empty((n_sec, n_chord_stations))
        z_lo = np.empty((n_sec, n_chord_stations))
        for i, row in enumerate(para):
            cu, cl = self.cst_rec(row, self.N1, self.N2, n_chord_stations)
            z_up[i] = cu[1]
            z_lo[i] = cl[1]

        # 2. 逐弦向站检查展向光顺度
        total_score = 0.0
        for j in range(n_chord_stations):
            for z_curve in [z_up[:, j], z_lo[:, j]]:
                cs = CubicSpline(y_span, z_curve)
                y_d = np.linspace(y_span[0], y_span[-1], 51)
                d2 = cs(y_d, 2)
                z_d = cs(y_d)

                rng = np.max(z_d) - np.min(z_d)
                if rng > 1e-8:
                    d2 = d2 / rng

                dy_d = np.diff(y_d)
                bend = float(np.sum(0.5 * (d2[:-1]**2 + d2[1:]**2) * dy_d))

                signs = np.sign(d2)
                nz = signs[np.abs(d2) > 1e-8]
                n_infl = int(np.sum(np.abs(np.diff(nz)) > 0)) if len(nz) > 1 else 0

                # 曲率总变差：惩罚曲率剧烈变化（鼓包/突变，无论拐点有无）
                tv_curv = float(np.sum(np.abs(np.diff(d2))))

                total_score += bend + n_infl**2 * 2.0 + tv_curv * 0.5

        return total_score

    def check_le_te(self) -> float:
        """前后缘展向曲线几何合理性检查

        约束：
        1. 前缘 le(y) 单调递增（后掠角不反向）
        2. 前缘 d²le/dy² ≤ 0（后掠角沿展向递减，物理合理）
        3. 后缘 te(y) 最多一次转折（允许类 baseline 的先减后增）
        4. 后缘 x ≤ 对称面后缘 x（te 不超出机身末端）
        5. 前后缘不交叉：te > le 处处成立（弦长 > 0）

        Returns:
            penalty: float, 0 = 全部满足, >0 = 违规程度
        """
        para = self.origin_para
        y = para[:, 0]
        le = para[:, -5]
        te = para[:, -4]

        penalty = 0.0

        # 1. 前缘单调递增
        dle = np.diff(le)
        viol = dle < 0
        if np.any(viol):
            penalty += np.sum(dle[viol] ** 2)

        # 2. 前缘 d²le/dy² ≤ 0（后掠角沿展向递减）
        # 使用离散二阶差分直接检查设计参数（dy=2 均匀）
        ddle = np.diff(le, n=2)
        viol = ddle > 1e-8
        if np.any(viol):
            penalty += np.sum(ddle[viol] ** 2)

        # 3. 后缘最多一次转折（dte 符号变化次数 ≤ 1）
        dte = np.diff(te)
        signs = np.sign(dte)
        nonzero = signs[signs != 0]
        n_turns = int(np.sum(np.abs(np.diff(nonzero)) > 0)) if len(nonzero) > 1 else 0
        if n_turns > 1:
            penalty += (n_turns - 1) * 10.0

        # 4. 后缘 x ≤ 对称面后缘 x
        over = te - te[0]
        if np.any(over > 0):
            penalty += np.sum(over[over > 0] ** 2)

        # 5. 前后缘不交叉: te > le
        gap = te - le
        if np.any(gap <= 0):
            penalty += np.sum(gap[gap <= 0] ** 2) * 100.0

        return penalty

    def check_z_offset(self, tol: float = 0.05) -> float:
        """z_offset 展向单调性检查（防止鼓包）

        z_offset 应沿展向单调递增（至少不显著下降），下降量超过 tol 即违规。

        Returns:
            penalty: float, 0 = 单调（或下降在容差内）, >0 = 违规程度
        """
        zo = self.origin_para[:, -3]
        dz = np.diff(zo)
        viol = dz < -tol
        if np.any(viol):
            return float(np.sum(dz[viol] ** 2))
        return 0.0

    def cal_areav_2(self, mach, aoa=0.0, n_span=40, n_chord=60, n_mach=200):
        """基于结构化网格截交线插值计算体积等效截面积 (全机)

        算法与 FABOOM slice 子程序等价:
        1. 坐标旋转 (迎角)
        2. 马赫面截距: X = x + z·√(M²-1)
        3. 各展向站弦向线性插值 → 交点 (y, z_u, z_l)
        4. S_V = 2·∫(z_u - z_l)dy   (对称加倍, 等价于多边形面积公式)
        """
        assert mach >= 1.0, "仅支持超声速 (Ma >= 1)"

        beta = np.sqrt(mach**2 - 1)  # = 1/tan(μ), Prandtl-Glauert 因子

        mesh = self.gene_simple_mesh(n_span, n_chord, aoa)

        # 马赫面截距: xty = x + z·β  (FABOOM: x + (z-H)·β, H偏移最终抵消)
        x_mach = mesh[..., 0] + mesh[..., 2] * beta  # [2, n_span, n_chord]

        X = np.linspace(x_mach.min(), x_mach.max(), n_mach)
        y_span = mesh[0, :, 0, 1]

        area = np.zeros(n_mach)
        bad_stations = 0
        for j in range(n_span):
            # 梯形积分权重
            if j == 0:
                w = 0.5 * (y_span[1] - y_span[0])
            elif j == n_span - 1:
                w = 0.5 * (y_span[j] - y_span[j - 1])
            else:
                w = 0.5 * (y_span[j + 1] - y_span[j - 1])

            # 弦向线性插值 → 马赫面与上下表面的 z 交点
            zu_k = np.interp(X, x_mach[0, j, :], mesh[0, j, :, 2],
                             left=np.nan, right=np.nan)
            zl_k = np.interp(X, x_mach[1, j, :], mesh[1, j, :, 2],
                             left=np.nan, right=np.nan)

            valid = ~np.isnan(zu_k) & ~np.isnan(zl_k)
            if valid.sum() < 2:
                bad_stations += 1
                continue

            dz = zu_k[valid] - zl_k[valid]
            # 上下表面交叉(畸形几何): 超过30%有效点 zu<zl 则跳过该站位
            if (dz < 0).sum() > 0.3 * valid.sum():
                bad_stations += 1
                continue

            area[valid] += dz * w * 2  # ×2 对称加倍

        # 超过一半展向站位异常 → 几何不合理, 返回 NaN 标记供调用方识别
        if bad_stations > n_span // 2:
            return np.array([[np.nan, np.nan]])

        return np.vstack([X, area]).T

    def cal_areav_panel(self, mach, aoa=0.0, n_mach=200,
                        n_span=100, n_chord=200):
        """基于 panel mesh 重采样 → 结构化网格 → 截交线插值计算 S_V(x) (全机)

        1. 收集 panel mesh 各域上/下表面点云
        2. LinearNDInterpolator 构建 z(x,y) 插值器
        3. 规则 (x,y) 网格上弦向插值 + 展向梯形积分 + 对称加倍
        """
        assert mach >= 1.0, "仅支持超声速 (Ma >= 1)"

        beta = np.sqrt(mach**2 - 1)  # Prandtl-Glauert 因子 = 1/tan(μ)

        panels = self.gene_panel_mesh(aoa)
        dom1, dom2 = panels[0], panels[1]
        dom3, dom4 = panels[2], panels[3]
        dom6, dom7 = panels[5], panels[6]
        dom9, dom10 = panels[8], panels[9]

        # 还原 gene_panel_mesh 翻转的下表面 j 方向
        dom2 = dom2[:, ::-1]
        dom4 = dom4[:, ::-1]
        dom7 = dom7[:, ::-1]

        # 收集点云
        pts_u = np.vstack([d.reshape(-1, 3) for d in [dom1, dom3, dom6, dom9]])
        pts_l = np.vstack([d.reshape(-1, 3) for d in [dom2, dom4, dom7, dom10]])

        # 展向网格 (半模 y≥0)
        y_min = max(pts_u[:, 1].min(), pts_l[:, 1].min())
        y_max = min(pts_u[:, 1].max(), pts_l[:, 1].max())
        y_grid = np.linspace(y_min, y_max, n_span)

        # 统一马赫面
        xm_all = np.concatenate([
            pts_u[:, 0] + pts_u[:, 2] * beta,
            pts_l[:, 0] + pts_l[:, 2] * beta])
        X = np.linspace(xm_all.min(), xm_all.max(), n_mach)

        area = np.zeros(n_mach)

        from scipy.interpolate import LinearNDInterpolator
        interp_u = LinearNDInterpolator(pts_u[:, :2], pts_u[:, 2])
        interp_l = LinearNDInterpolator(pts_l[:, :2], pts_l[:, 2])

        for j in range(n_span):
            if j == 0:
                w = 0.5 * (y_grid[1] - y_grid[0])
            elif j == n_span - 1:
                w = 0.5 * (y_grid[j] - y_grid[j - 1])
            else:
                w = 0.5 * (y_grid[j + 1] - y_grid[j - 1])

            yj = y_grid[j]

            # 该展向站的有效弦向范围
            tol = (y_grid[1] - y_grid[0]) * 2
            mu_mask = np.abs(pts_u[:, 1] - yj) < tol
            ml_mask = np.abs(pts_l[:, 1] - yj) < tol
            if mu_mask.sum() < 3 or ml_mask.sum() < 3:
                continue

            x_min = max(pts_u[mu_mask, 0].min(), pts_l[ml_mask, 0].min())
            x_max = min(pts_u[mu_mask, 0].max(), pts_l[ml_mask, 0].max())
            if x_max <= x_min:
                continue
            xi = np.linspace(x_min, x_max, n_chord)

            # 2D 插值 → 弦向截面 z(x)
            zu_xi = interp_u(xi, np.full(n_chord, yj))
            zl_xi = interp_l(xi, np.full(n_chord, yj))

            ok = ~np.isnan(zu_xi) & ~np.isnan(zl_xi)
            if ok.sum() < 2:
                continue
            xi, zu_xi, zl_xi = xi[ok], zu_xi[ok], zl_xi[ok]

            # 弦向 → 马赫面插值: X = x + z·β
            xu_m = xi + zu_xi * beta
            xl_m = xi + zl_xi * beta
            zu_k = np.interp(X, xu_m, zu_xi, left=np.nan, right=np.nan)
            zl_k = np.interp(X, xl_m, zl_xi, left=np.nan, right=np.nan)
            valid = ~np.isnan(zu_k) & ~np.isnan(zl_k)
            area[valid] += (zu_k[valid] - zl_k[valid]) * w * 2  # ×2 对称加倍

        return X, area

    def search_cabin(self):
        mesh = self.air_mesh.reshape([-1, 3])
        total_p = mesh.shape[0]
        chaos = 0
        ## 定义客舱尺寸，展向有微量收缩作为容差
        cabin_top = 1.55; cabin_height = 2.0
        cabin_begin = 20; cabin_len = 40*0.95; cabin_half_width = (4*0.55+0.5)/2; tol = 0.05
        cabin_end = cabin_begin + cabin_len; cabin_ground = cabin_top - cabin_height
        for point in mesh:
            if (point[0] > cabin_begin) and (point[0] < cabin_end):
                if (point[1] < cabin_half_width):
                    l_edge = cabin_ground + tol * (point[1]/cabin_half_width)
                    u_edge = cabin_top - tol * (point[1]/cabin_half_width)
                    if (point[2] < u_edge) and (point[2] > l_edge):
                        chaos += 1

        return chaos/total_p
        
    def ref_area(self):
        para = self.origin_para
        area = 0
        for i in range(para.shape[0]-1):
            c1, c2 = (para[i+1, -4] - para[i+1, -5]), (para[i, -4] - para[i, -5])
            dy = para[i+1, 0] - para[i, 0]
            area += 0.5 * (c1 + c2) * dy

        return area

if __name__ == "__main__":
    bwb0 = Aircraft()
    bwb0.read_from_csv(r"baseline_6tip.csv")
    print(bwb0.if_smooth())
    print(bwb0.search_cabin())
    print(bwb0.tail_new())
    print(bwb0.ref_area())
    bwb0.write_mesh("simple", "test.x", 0)
    bwb0.write_mesh("panel", r"FABOOM_test\indata\geo.x", 0)
    print(cal_Lift())

    # plt.axis('equal')
    plt.show()