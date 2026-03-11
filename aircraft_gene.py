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
    
    def interp_para(self, num_span) -> ndarray: #补全对称条件并插值参数列表
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
        full_para = np.empty([1, self.origin_para.shape[1]])
        full_para[0] = y

        for j in range(self.origin_para.shape[1]-1):
            full_mesh_para = np.append(ori_para[:,j+1][1:][::-1], ori_para[:,j+1])
            
            if j == ori_para.shape[1]-5: #后缘保持一阶连续
                f = si.interp1d(full_y_list, full_mesh_para, kind='linear')
            else:
                f = si.interp1d(full_y_list, full_mesh_para, kind='quadratic')
            para = f(y)
            full_para[0, j+1] = para

        return full_para
    
    def cst_rec(self, coeffs, cst_order, le, te, z_offset, dy_upper=0, dy_lower=0, N1=0.5, N2=1, n_points=60, psi_end=1.0):
        psi = np.linspace(0, psi_end, n_points)
        coeffs_upper = coeffs[0]
        coeffs_lower = coeffs[1]
        
        # 生成Bernstein基函数
        B = np.zeros((n_points, cst_order+1))
        for i in range(cst_order+1):
            B[:, i] = comb(cst_order, i) * (psi**i) * (1 - psi)**(cst_order-i)
        
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
            cst = np.array([data[1:order+2],data[order+2:(order+1)*2+1]])
            coord_u, coord_l = self.cst_rec(cst, order, data[-5], data[-4], data[-3], data[-2], data[-1], self.N1, self.N2, num_chord)
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
            L_inv = si.interp1d(L, xtmp, kind=2)
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
        dom6, dom7 = np.zeros([wing_i, wing_j-1, 3]), np.zeros([wing_i, wing_j-1, 3]) ##机翼网格
        dom8 = np.zeros([body_i, 2, 3]) ##翼尖网格
        dom9, dom10 = np.zeros([tail_i, tail_j, 3]), np.zeros([tail_i, tail_j, 3]) ##尾部网格
        dom11 = np.zeros([tail_j, 2, 3]) ##钝底网格
        dom12 = np.zeros([tail_i+1, 2, 3]) ##尾涡面网格

        ##机头网格计算
        #===================================#
        leading_edge_x = this_para[:, 2*order+3]
        leading_edge_z = this_para[:, 2*order+5]
        f_leading_xy = si.interp1d(leading_edge_x, this_para[:, 0], kind=2)
        f_leading_xz = si.interp1d(leading_edge_x, leading_edge_z, kind=2)
        leading_deri = deri_1d(leading_edge_x, this_para[:, 0])
        mask = (leading_edge_x > 3)&(leading_deri > 0.15)
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
                cst = np.array([da[1:order+2],da[order+2:(order+1)*2+1]])
                z_u, z_l = self.cst_rec(cst, order, da[-5], da[-4], da[-3], da[-2], da[-1], self.N1, self.N2, 2, psi_end)
                coords_this[idx, 2] = z_u[1, -1]
                coords_this[idx, 3] = z_l[1, -1]
            coords_this[-1, 2] = this_z_end
            coords_this[-1, 3] = this_z_end
            new_coords = np.ones([nose_j, 3]) * x
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 2], nose_j)
            dom1[i+1] = new_coords
            new_coords[:, 1], new_coords[:, 2] = redistribution(coords_this[:, 1], coords_this[:, 3], nose_j)
            dom2[i+1] = new_coords
        #===================================#

        ##机身网格计算
        #===================================#
        ##从后缘曲线截取与机头结束y值相等的x值，确定x范围
        x_begin = dom1_end
        trailing_edge_x = this_para[:, -4]
        f_trailing_xy = si.interp1d(this_para[:, 0], trailing_edge_x, kind=2)
        x_end = f_trailing_xy(this_y_end)
        print(x_begin, x_end)
        x_list = np.linspace(x_begin, x_end, body_i)
        for i, x in enumerate(x_list):
            pass
        #===================================#
        
        panel_mesh = [locals()[f"dom{i}"] for i in range(1, 12)] ##由于分块网格长度尺度不统一用列表存储

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
                    dom = dom.transpose(2, 0, 1).flatten().reshape([-1, 5])
                    for line in dom:
                            f.write(" ".join(f"{x:.6f}" for x in line) + "\n")
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
    air_para.write_mesh(test_mesh, "geo.x")

    plt.show()