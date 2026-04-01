import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import test
from aircraft_gene import Aircraft

class aircraft_para:
    def __init__(self, origin_X) -> None:
        self.origin_X = origin_X
        self.n_sample = self.origin_X.shape[0]
        self.X = origin_X.reshape(self.n_sample, -1) if (len(origin_X.shape) > 2) else origin_X
        self.mean = np.mean(self.X, axis=0)
        self.X_center = self.X - self.mean

    def PCA(self, k):
        mean = self.mean
        X_center = self.X_center
        Cov = X_center.T @ X_center / (self.n_sample - 1)
        U, s = np.linalg.eigh(Cov)
        U = U[::-1] #奇异值
        s = s[:, ::-1] #主模态
        ## 取前k个主成分
        s_k = s[:, :k]
        Z = X_center @ s_k
        Z_min = np.min(Z, axis=0)  # (k,)
        Z_max = np.max(Z, axis=0)  # (k,)
        test_Z = np.random.uniform(Z_min, Z_max)
        X_recon = test_Z @ s_k.T + mean
        X_recon = X_recon.reshape(-1, 24)
        print(X_recon.shape)
        air_test = Aircraft(X_recon)
        air_test.write_mesh("panel", 'check.x', 0)
        print(X_recon.shape)
        plt.plot(list(range(len(U))), U)

    def kernel_PCA(self, k):
        mean = self.mean
        X = self.X
        X_center = self.X_center
        n_sample = self.n_sample

        ## RBF核矩阵
        gamma = 0.1 #超参数, ai推荐0.01~0.1
        K = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)

        one_mat = np.ones((n_sample, n_sample))
        K_center = K - one_mat @ K - K @ one_mat + one_mat @ K @ one_mat

        eigen_vals, eigen_vecs = np.linalg.eigh(K_center)
        eigen_vals = eigen_vals[::-1]         # 特征值从大到小
        eigen_vecs = eigen_vecs[:, ::-1]      # 特征向量对应排序

        plt.plot(list(range(len(eigen_vals))), eigen_vals)
        plt.yscale('log')

        eigen_vals[eigen_vals < 1e-6] = 1e-6
        alpha = eigen_vecs[:, :k] / np.sqrt(eigen_vals[:k])
        Z_all = K_center @ alpha
        Z_min = np.min(Z_all, axis=0)
        Z_max = np.max(Z_all, axis=0)
        test_Z = np.random.uniform(Z_min, Z_max).reshape(1, -1)
        print(X_center.shape)

        weights = test_Z @ alpha.T  # (1, n_sample)
        X_recon = mean + weights @ X_center
        
        X_recon = X_recon.reshape(-1, 24)  # 改成你的网格形状

        air_test = Aircraft(X_recon)
        air_test.write_mesh("panel", 'kernel_pca_test.x')
        print("Kernel PCA 生成完成！shape =", X_recon.shape)
        PCA_dict = {}
        PCA_dict["alpha"] = alpha
        PCA_dict["X_center"] = X_center
        PCA_dict["mean"] = mean
        PCA_dict["Z_min"] = Z_min
        PCA_dict["Z_max"] = Z_max
        np.savez("PCA_dict.npz", **PCA_dict)

if __name__ == "__main__":
    data_path = "no_lift_samples.csv"
    data = pd.read_csv(data_path).to_numpy()[:1000, 2:]
    para = aircraft_para(data)
    k = 15
    para.kernel_PCA(k)

    plt.show()