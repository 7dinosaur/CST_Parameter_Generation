import numpy as np
from aircraft_gene import Aircraft
from cal_Lift import cal_Lift, cal_PLdB

def opt_fun_PLdB(var):  # 输入低维变量
    PCA_dict = np.load("PCA_dict.npz", allow_pickle=True)
    alpha = PCA_dict["alpha"]
    mean = PCA_dict["mean"]
    X_center = PCA_dict["X_center"]
    Z_min = PCA_dict["Z_min"]
    Z_max = PCA_dict["Z_max"]
    var = (Z_min + var * (Z_max - Z_min)).reshape(1, -1)

    # ===== 1. kernel PCA 逆映射 =====
    weights = var @ alpha.T
    X_recon = mean + weights @ X_center
    X_recon = X_recon.reshape(-1, 24)

    # ===== 2. 构造飞机 =====
    air = Aircraft(X_recon)

    # ===== 3. 几何约束 =====

    passenger = air.cal_volume() - 140
    if passenger < 0:
        return 1e6, -1, -1

    # ===== 4. 写网格 =====
    air.write_mesh("panel", r"FABOOM_test\indata\geo.x", 3.8)

    # ===== 5. 气动 =====
    Lift = cal_Lift()
    if Lift is False:
        return 1e6, -1, -1
    Lift = Lift - 1190000
    if Lift < 0:
        return 1e6, -1, -1

    # ===== 6. 声爆 =====
    PLdB = cal_PLdB()

    return PLdB, Lift - 1190000, passenger - 140

# ======== 测试 ========
if __name__ == "__main__":
    x = np.random.rand(10)
    print(opt_fun_PLdB(x))