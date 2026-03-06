import numpy as np
import pandas as pd
from numpy.typing import NDArray
import scipy.interpolate as si
from scipy.special import comb
from matplotlib import pyplot as plt

class Aircraft:
    def __init__(self, para_file : str) -> None:
        self.origin_para = self.read_csv(para_file)
        self.cst_order = int(0.5 * (self.origin_para.shape[1] - 8))
        self.N1 = 0.5
        self.N2 = 1
        self.air_mesh:NDArray = np.array([0])
        self.panel_mesh:NDArray = np.array([0])

    def read_csv(self, csv_file : str) -> NDArray:
        mesh_para = pd.read_csv(csv_file).to_numpy()
        return mesh_para
    
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
    
    def gene_panel_mesh(self, para) -> None:
        """生成三维网格数组,第一维为dom编号,如aircraft[0]=dom1,二三维为ij方向,四维[x,y,z]"""
        """panel_mesh是可以直接输入faboom程序计算的分块网格"""
    
    def write_mesh(self, mesh:NDArray | list, file_path:str) -> None:
        with open(file_path, 'w') as f:
            if type(mesh) == np.ndarray: #simple mesh
                print("写入一般网格")
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

    
class Aircraft_generator:
    def __init__(self, ) -> None:
        pass

if __name__ == "__main__":
    air_para = Aircraft("increase_cabin.csv")
    air_para.gene_simple_mesh(41, 60)
    test_mesh = air_para.air_mesh
    air_para.write_mesh(test_mesh, "test_mesh.x")
    plt.plot(test_mesh[0, 0, :, 0], test_mesh[0, 0, :, 2])
    plt.plot(test_mesh[1, 0, :, 0], test_mesh[1, 0, :, 2])
    plt.show()