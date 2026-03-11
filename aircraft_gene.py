import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numpy import ndarray
import scipy.interpolate as si
from scipy.special import comb
from matplotlib import pyplot as plt

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

class Aircraft:
    def __init__(self) -> None:
        self.origin_para:ndarray = np.array([0])
        self.cst_order:int = 0
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
            
            if j == ori_para.shape[1]-5: #后缘保持一阶连续
                f = si.interp1d(full_y_list, full_mesh_para, kind='linear')
            else:
                f = si.interp1d(full_y_list, full_mesh_para, kind='quadratic')
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
            
            if j == ori_para.shape[1]-5: #后缘保持一阶连续
                f = si.interp1d(full_y_list, full_mesh_para, kind='linear')
            else:
                f = si.interp1d(full_y_list, full_mesh_para, kind='quadratic')
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

    def gene_simple_mesh(self, num_span, num_chord) -> None:
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
    
    def gene_panel_mesh(self) -> list[ndarray]:
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
        leading_edge_z = this_para[:, 2*order+5]
        f_leading_xy = si.interp1d(leading_edge_x, this_para[:, 0], kind='quadratic')
        f_leading_xz = si.interp1d(leading_edge_x, leading_edge_z, kind='quadratic')
        leading_deri = deri_1d(leading_edge_x, this_para[:, 0])
        mask = (leading_edge_x > 3)&(leading_deri > 0.12)
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
        test = trailing_edge_x
        for i, t in enumerate(test):
            if test[i+1] > test[i]:
                print(i)
                break
        trailing_edge_x = test[:i]; trailing_edge_y = this_para[:, 0][:i]
        f_trailing_xy = si.interp1d(trailing_edge_x, trailing_edge_y, kind='quadratic')
        for i, x in enumerate(x_list[:-1]):
            this_y_end = f_trailing_xy(x)
            this_line = self.interp_single_para(this_y_end)
            mask = this_para[:, 0] < this_y_end - 0.1*delta_y
            tmp_para = this_para[mask].copy() #获得从对称面到结束位置的参数
            tmp_para = np.append(tmp_para, this_line.reshape([1, -1]), axis=0)
            coords_this = np.zeros([tmp_para.shape[0], 4])
            coords_this[:, 0] = x
            coords_this[:, 1] = tmp_para[:, 0]
            for idx, da in enumerate(tmp_para):
                psi_end = (x - tmp_para[idx, -5])/(tmp_para[idx, -4] - tmp_para[idx, -5])
                z_u, z_l = self.cst_rec(da, self.N1, self.N2, 2, psi_end)
                coords_this[idx, 2] = z_u[1, -1]
                coords_this[idx, 3] = z_l[1, -1]
            new_coords = np.ones([tail_j, 3]) * x
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 2], tail_j)
            dom9[i] = new_coords
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 3], tail_j)
            dom10[i] = new_coords
        dom10 = dom10[:, ::-1]
        end_z = this_para[0, -3] + (this_para[0, -4] - this_para[0, -5])*this_para[0, -1]
        dom9[-1, :] = dom10[-1, :] = np.array([this_para[0, -4], this_para[0, 0], end_z])
        print(this_para[0, 0])
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
        
        panel_mesh = [locals()[f"dom{i}"] for i in range(1, 13)] ##由于分块网格长度尺度不统一用列表存储
        panel_mesh.pop(-2)

        return panel_mesh
    
    def write_mesh(self, mesh:NDArray | list, file_path:str) -> None:
        """自动识别网格类型并写入文件"""
        with open(file_path, 'w') as f:
            if type(mesh) == ndarray: #simple mesh
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

            elif type(mesh) == list: #panel mesh
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
        height_cabin = 2 #客舱高度为2m
        weight_cabin = 3.6 #客舱宽度为3.6m
        half_weight = 0.5 * weight_cabin
        para = self.interp_para(51)
        for spara in para:
            pass
    
class Aircraft_generator:
    def __init__(self, ) -> None:
        pass

if __name__ == "__main__":
    air_para = Aircraft()
    air_para.read_from_csv("increase_cabin.csv")
    # air_para.gene_simple_mesh(41, 60)
    test_mesh = air_para.gene_panel_mesh()
    # air_para.write_mesh(test_mesh, "test_mesh.x")
    air_para.write_mesh(test_mesh, "FABOOM_test\\indata\\geo.x")

    plt.show()