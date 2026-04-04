from heapq import nsmallest

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from aircraft_gene import Aircraft

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'

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
        ## 协方差矩阵
        Cov = X_center.T @ X_center / (self.n_sample - 1)
        eigen_vals, eigen_vecs = np.linalg.eigh(Cov)
        eigen_vals = eigen_vals[::-1]         # 特征值从大到小
        eigen_vecs = eigen_vecs[:, ::-1]      # 特征向量对应排序

        total_energy = np.sum(eigen_vals)
        explained_ratio = eigen_vals / total_energy
        cumulative_energy = np.cumsum(explained_ratio)

        print("======= PCA 能量信息 =======")
        print(f"总能量: {total_energy:.2e}")
        print(f"前{k}个主成分累计能量: {cumulative_energy[k-1]:.2%}")
        print(f"前10个能量: {cumulative_energy[9]:.2%}")
        print(f"前20个能量: {cumulative_energy[19]:.2%}")

        # ==========================
        # ✅ 画图
        # ==========================
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(explained_ratio[:50], 'o-')
        plt.title('每个主成分贡献率')
        plt.ylabel('贡献率')
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(cumulative_energy[:50], 'r-')
        plt.title('累计贡献率')
        plt.ylabel('累计能量')
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)

        ## 取前k个主成分
        s_k = eigen_vecs[:, :k]
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
        # plt.plot(list(range(len(U))), U)

    def kernel_PCA(self, k):
        mean = self.mean
        X = self.X
        X_center = self.X_center
        n_sample = self.n_sample

        ## RBF核矩阵
        gamma = 0.001 #超参数, ai推荐0.01~0.1
        K = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)

        one_mat = np.ones((n_sample, n_sample)) / n_sample
        K_center = K - one_mat @ K - K @ one_mat + one_mat @ K @ one_mat

        eigen_vals, eigen_vecs = np.linalg.eigh(K_center)
        eigen_vals = eigen_vals[::-1]         # 特征值从大到小
        eigen_vecs = eigen_vecs[:, ::-1]      # 特征向量对应排序
        mask = eigen_vals > 1e-6
        eigen_vals = eigen_vals[mask]
        eigen_vecs = eigen_vecs[:, mask]

        total_energy = np.sum(eigen_vals)
        explained_ratio = eigen_vals / total_energy  # 每个主成分的贡献率
        cumulative_energy = np.cumsum(explained_ratio)  # 累计贡献率

        print("总能量:", total_energy)
        print("前10个主成分累计能量:", cumulative_energy[:10])
        print("前20个主成分累计能量:", cumulative_energy[:20])

        
        
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

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(explained_ratio[:100], 'o-')
        plt.title('每个主成分贡献率')
        plt.subplot(1,2,2)
        plt.plot(cumulative_energy[:100], 'r-')
        plt.title('累计贡献率')

        PCA_dict = {}
        PCA_dict["alpha"] = alpha
        PCA_dict["X_center"] = X_center
        PCA_dict["mean"] = mean
        PCA_dict["Z_min"] = Z_min
        PCA_dict["Z_max"] = Z_max
        np.savez("PCA_dict_opt1.npz", **PCA_dict)

if __name__ == "__main__":
    data_path = "samples_based_opt1_test_highrate.csv"
    data = pd.read_csv(data_path).to_numpy()[:1000, 2:]
    para = aircraft_para(data)
    k = 15
    para.kernel_PCA(k)
    # para.PCA(k)

    plt.show()