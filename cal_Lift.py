import os
import subprocess
import numpy as np
import shutil

def cal_Lift(base_path : str = "FABOOM_test") -> bool | float:
    base_dir = os.path.join(os.path.dirname(__file__), base_path)
    exe_path = os.path.join(base_dir, r"FABOOM.exe") #拼接程序执行路径
    result_path = os.path.join(base_dir, r"A502\\Lift distribution.dat")
    
    try:
        # 异步执行FABOOM程序
        print(f"正在执行气动计算，目录：{base_dir}")
        process = subprocess.Popen([exe_path], cwd=base_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 等待外部程序执行完成并获取结果
        stdout, stderr = process.communicate()

        # 错误判断
        fail_keywords = ["forrt", "error", "Unknown"]
        output = stdout + stderr
        calc_success = not any(key in output for key in fail_keywords)
        
        if calc_success:
            lift = np.loadtxt(result_path)
            print(lift[-1, 1])
            return lift[-1, 1]
        else:
            print(output)
            return False

    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def cal_PLdB() -> float:
    shutil.copy2("FABOOM_test\\outdata\\nearfield.dat", "FABOOM_test\\bboom\\INDATA\\near_field_pressure.dat")
    base_dir = os.path.join(os.path.dirname(__file__), "FABOOM_test\\bboom")
    exe_path = os.path.join(base_dir, r"bBoom.exe") #拼接程序执行路径
    result_path = os.path.join(base_dir, r"OUTDATA\\Burgers_NOISE.DAT")
    
    try:
        # 调用bBoom程序
        print("正在执行声爆传播计算")
        result = subprocess.run([exe_path], cwd=base_dir, check=True, text=True, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        val = float([*open(result_path)][3].split()[2])
        return val
    
    except Exception as e:
        print(f"错误: {str(e)}")
        return -1.0
    
if __name__ == "__main__":
    # cal_Lift()
    print(cal_PLdB())