import numpy as np
import pandas as pd
from numpy.typing import NDArray
import scipy.interpolate as si

class Aircraft_parameter:
    def __init__(self, para_file : str) -> None:
        self.origin_para = self.read_csv(para_file)
        self.cst_order = int(0.5 * (self.origin_para.shape[1] - 8))

    def read_csv(self, csv_file : str) -> NDArray:
        mesh_para = pd.read_csv(csv_file).to_numpy()
        return mesh_para
    
    def interp_para(self, num_span) -> None: #补全对称条件并插值参数列表
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
        self.interped_para = full_para

    

if __name__ == "__main__":
    air_para = Aircraft_parameter("increase_cabin.csv")
    air_para.interp_para(41)
    print(air_para.interped_para.shape)
    print(air_para.cst_order)