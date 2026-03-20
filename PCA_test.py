import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from aircraft_gene import Aircraft

class aircraft_para:
    def __init__(self, origin_X) -> None:
        self.origin_X = origin_X
        self.n_sample = self.origin_X.shape[0]
        self.X = origin_X.reshape(self.n_sample, -1) if (len(origin_X.shape) > 2) else origin_X

    def PCA(self, k):
        mean = np.mean(self.X, axis=0)
        X_center = self.X - mean
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

    def kernel_PCA(self):
        pass

if __name__ == "__main__":
    data_path = "qualified_solutions_1_set.csv"
    data = pd.read_csv(data_path).to_numpy()[:, 3:]
    para = aircraft_para(data)
    k = 5
    para.PCA(k)
    plt.show()