import os
import subprocess
import numpy as np
import pandas as pd

from aircraft_gene import Aircraft

def cal_Lift() -> bool:
    base_dir = os.path.join(os.path.dirname(__file__), "FABOOM_test")
    exe_path = os.path.join(base_dir, r"FABOOM.exe") #拼接程序执行路径
    result_path = os.path.join(base_dir, r"A502\\Lift distribution.dat")
    
    try:
        # 调用FABOOM程序
        print("正在执行气动计算")
        result = subprocess.run([exe_path], cwd=base_dir, check=True, text=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        ##错误判断，貌似没用，不过还是留着吧
        #===============================================================#
        output = result.stdout + result.stderr
        fail_keywords = ["forrt", "error", "Unknown"]
        calc_success = not any(key in output for key in fail_keywords)
        if calc_success:
            lift = np.loadtxt(result_path)
            print(lift[-1, 1])
            return calc_success
        else:
            print(output)
            return False
        #===============================================================#
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False
    
def perturb_para(base_para, perturbation=0.05):
    new_para = base_para.copy()
    params_to_perturb = new_para[:, 1:17]
    
    perturb_factor = 1 + perturbation * (2 * np.random.rand(*params_to_perturb.shape) - 1)
    params_to_perturb = params_to_perturb * perturb_factor
    new_para[:, 1:17] = params_to_perturb

    return new_para

def main():
    para_csv = "increase_cabin.csv"
    base_para = pd.read_csv(para_csv).to_numpy()
    new_para = perturb_para(base_para, 0.001)
    new_air = Aircraft(new_para)
    new_air.write_mesh("panel", r"FABOOM_test\indata\geo.x")

    print(cal_Lift())

if __name__ == "__main__":
    main()