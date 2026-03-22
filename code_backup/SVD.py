import pandas as pd
import numpy as np
from numpy import ndarray
from aircraft_gene import Aircraft
from matplotlib import pyplot as plt

def SVD(X:ndarray, k):
    # 展平
    X = X.reshape(X.shape[0], -1) if (len(X.shape) > 2) else X
    # 中心化
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # 分解
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    plt.plot(*list(range(1, len(s)+1)), s)
    plt.ylim(0, 1)
    U_k, s_k, Vt_k = U[:, :k], s[:k], Vt[:k, :]
    X_reduced = U_k @ np.diag(s_k)  # 降维结果
    print(s)
    coef = 1
    var_min, var_max = -coef*s_k, +coef*s_k
    # print(f"变量下限：{var_min}, 变量上限：{var_max}")
    # 重构
    X_recon_centered = X_reduced @ Vt_k  # 回到 192 维，但中心化
    X_recon = X_recon_centered + mean    # 加回均值
    print(X_recon[0].reshape(-1, 24)[:, 0])

    error = np.mean((X - X_recon) ** 2)
    print(error)
    return X_reduced, Vt_k, mean, var_min, var_max

def SVD_recon(var, Vt_k, mean):
    X_recon_centered = var @ Vt_k  # 回到 192 维，但中心化
    X_recon = X_recon_centered + mean    # 加回均值
    return X_recon

if __name__ == "__main__":
    data_path = "qualified_solutions_1.csv"
    data = pd.read_csv(data_path).to_numpy()[:, 3:] # 每行代表一个样本的飞机参数, 如(14,192)代表14个样本
    k = 3
    X_reduced, Vt_k, mean, var_min, var_max = SVD(data, k)
    var_test = X_reduced[0]
    recon_sample = SVD_recon(var_test, Vt_k, mean)
    new_var = var_min + np.random.rand(k) * (var_max - var_min)
    recon_sample = SVD_recon(new_var, Vt_k, mean)
    para = recon_sample.reshape(-1, 24)
    test_air = Aircraft(para)
    test_air.write_mesh("panel", "check.x", 0)
    plt.show()