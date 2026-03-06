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
    
    def interp_para(self, num_span) -> NDArray: #补全对称条件并插值参数列表
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
    
    def cst_rec(self, coeffs, cst_order, le, te, z_offset, dy_upper=0, dy_lower=0, N1=0.5, N2=1, n_points=60):
        psi = np.linspace(0, 1, n_points)
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

        this_para = self.interp_para(61) ##此处插值仅保证曲线光滑，实际网格尺度与插值长度无关
        order = self.cst_order

        ##统一网格尺度设置
        nose_i, body_i, tail_i = 31, 60, 10
        nose_j = body_j = tail_j = 10
        wing_i = body_i
        wing_j = 29

        ##头部网格计算
        dom1, dom2 = np.zeros([nose_i, nose_j]), np.zeros([nose_i, nose_j]) ##机头网格
        dom3, dom4 = np.zeros([body_i, body_j]), np.zeros([body_i, body_j]) ##机身网格
        dom5 = np.zeros([body_i, 2]) ##机翼上表面与尾涡面相连的网格
        dom6, dom7 = np.zeros([wing_i, wing_j-1]), np.zeros([wing_i, wing_j-1]) ##机翼网格
        dom8 = np.zeros([body_i, 2]) ##翼尖网格
        dom9, dom10 = np.zeros([tail_i, tail_j]), np.zeros([tail_i, tail_j]) ##尾部网格
        dom11 = np.zeros([tail_j, 2]) ##钝底网格
        dom12 = np.zeros([tail_i+1, 2]) ##尾涡面网格

        ##机头网格计算
        #===================================#
        leading_edge = this_para[:, 2*order+3]
        leading_deri = deri_1d(leading_edge, this_para[:, 0])
        # plt.plot(leading_edge, this_para[:, 0])
        plt.plot(leading_edge, leading_deri)
        #===================================#

        ##机身网格计算
        #===================================#
        #===================================#
        
        panel_mesh = [locals()[f"dom{i}"] for i in range(1, 12)] ##由于分块网格长度尺度不统一用列表存储

        return panel_mesh
    
    def write_mesh(self, mesh:NDArray | list, file_path:str) -> None:
        with open(file_path, 'w') as f:
            if type(mesh) == np.ndarray: #simple mesh
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
                    n_i, n_j = dom.shape[0], dom.shape[1]
                    f.write(f"{n_i} {n_j} 1\n")
    
class Aircraft_generator:
    def __init__(self, ) -> None:
        pass

if __name__ == "__main__":
    air_para = Aircraft()
    air_para.read_from_csv("increase_cabin.csv")
    # air_para.gene_simple_mesh(41, 60)
    test_mesh = air_para.gene_panel_mesh()
    # test_mesh = air_para.air_mesh
    air_para.write_mesh(test_mesh, "test_mesh.x")
    plt.show()