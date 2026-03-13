import os
import subprocess
import numpy as np

def cal_Lift() -> bool | float:
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
            return lift[-1, 1]
        else:
            print(output)
            return False
        #===============================================================#
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return False
    
if __name__ == "__main__":
    cal_Lift()