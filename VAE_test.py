import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from aircraft_gene import Aircraft

# ==============================================
# VAE 模型定义（固定不动）
# ==============================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        self.num_station = 7
        self.station_dim = 23
        self.latent_dim = latent_dim
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 128), nn.ReLU(),
        #     nn.Linear(128, 64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.ReLU(),
        #     nn.Linear(64, 64), nn.ReLU(),
        # )

        self.station_encoder = nn.Sequential(
            nn.Linear(self.station_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(16 * self.num_station, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 16 * self.num_station), nn.ReLU(), 
        )

        self.station_decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, self.station_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.num_station, self.station_dim)

        station_feats = []
        for i in range(self.num_station):
            xi = x[:, i, :]
            hi = self.station_encoder(xi)
            station_feats.append(hi)
        
        h = torch.cat(station_feats, dim=1)
        h = self.fusion_encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)
        h_dec = self.decoder(z)
        h_dec = h_dec.view(B, self.num_station, 16)

        recon_stations = []
        for i in range(self.num_station):
            hi = h_dec[:, i, :]
            xi = self.station_decoder(hi)
            recon_stations.append(xi)

        recon_x = torch.cat(recon_stations, dim=1)

        return recon_x, mu, logvar

# ==============================================
# 飞行器降维类（可保存、可加载、可直接输入0~1输出参数）
# ==============================================
class AircraftVAE:
    def __init__(self, latent_dim=10):
        self.latent_dim = latent_dim
        self.model = None
        self.x_min = None
        self.x_max = None
        self.x_mean = None
        self.input_dim = None

    # ---------------------
    # 训练 + 保存模型
    # ---------------------
    def train_and_save(self, data, model_path="vae_model.pth", epochs=800, lr=1e-3):
        self.input_dim = data.shape[1]
        self.x_min = np.min(data, axis=0)
        self.x_max = np.max(data, axis=0)
        self.x_mean = np.mean(data, axis=0)
        self.x_std = np.std(data, axis=0)
        # X_norm = (data - self.x_min) / (self.x_max - self.x_min + 1e-8)
        X_norm = (data - self.x_mean) / (self.x_std + 1e-6)
        x_tensor = torch.FloatTensor(X_norm)
        
        ## 添加batch
        dataset = torch.utils.data.TensorDataset(x_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        model = VAE(self.input_dim, self.latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        mse = nn.MSELoss()

        losses = []  # 就加这行

        print(f"训练VAE，隐变量维度：{self.latent_dim}")
        for epoch in range(epochs):
            total_loss = 0
            total_mse = 0
            total_kl = 0

            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x)
                recon_loss = mse(recon_x, x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                beta = min(1.0, epoch / 200)  # KL annealing
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_mse += recon_loss.item()
                total_kl += kl_loss.item()
            
            avg_loss = total_loss / len(loader)
            avg_mse = total_mse / len(loader)
            avg_kl = total_kl / len(loader)
            losses.append(avg_loss)  # 就加这行
            if epoch % 100 == 0:
                print(f"Epoch {epoch:3d} | Loss={avg_loss:.5f}, MSE={avg_mse:.5f}, KL={avg_kl:.5f}")

        # 保存模型 + 归一化参数
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "x_mean": self.x_mean,
            "x_std" : self.x_std
        }, model_path)
        self.model = model
        print("模型训练完成，已保存到：", model_path)

        plt.plot(losses)
        plt.yscale('log')
        plt.show()

    # ---------------------
    # 加载模型（以后直接用）
    # ---------------------
    def load_model(self, model_path="vae_model.pth"):
        ckpt = torch.load(model_path, weight_only=False)
        self.input_dim = ckpt["input_dim"]
        self.latent_dim = ckpt["latent_dim"]
        self.x_min = ckpt["x_min"]
        self.x_max = ckpt["x_max"]

        self.model = VAE(self.input_dim, self.latent_dim)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print("模型加载成功！")

    # ---------------------
    # 核心函数：输入 0~1 的10个数 → 输出飞行器参数
    # ---------------------
    def decode(self, z_01):
        """
        输入：z_01 = 10个0~1之间的数，shape=(10,)
        输出：飞行器设计参数，shape=(N, 24) 可直接写网格
        """
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z_01).unsqueeze(0)
            x_norm = self.model.decoder(z_tensor).numpy()[0]

        # 反归一化 → 真实尺度
        x_recon = x_norm * self.x_std + self.x_mean
        return x_recon.reshape(-1, 24)

# ==============================================
# 主程序：训练一次 → 以后永久调用
# ==============================================
if __name__ == "__main__":
    # 1. 读取数据
    data_path = "no_lift_samples.csv"
    data = pd.read_csv(data_path).to_numpy()[:1000, 2:].reshape([1000, -1, 24])
    first_col = data[0, :, 0]
    data = data[:, :, 1:].reshape([1000, -1])


    # 2. 创建VAE对象（10维隐空间）
    vae = AircraftVAE(latent_dim=30)

    # 3. 训练并保存模型（只需要运行一次！）
    vae.train_and_save(data, epochs=1000)

    # ==========================================
    # 以下是你未来优化程序要用到的代码
    # ==========================================

    # 4. 加载模型（以后直接用这行）
    # vae.load_model()

    # 5. 输入：10个 0~1 的随机数
    # z_random = -1.0 + 2 * np.random.rand(10)
    # print("输入VAE的10个随机数：\n", np.round(z_random, 3))

    # # 6. 解码 → 输出飞行器设计参数
    # aircraft_params = vae.decode(z_random)

    # # 7. 输出给你的优化程序
    # print("输出飞行器参数 shape：", aircraft_params.shape)

    # # 8. 直接写网格文件
    # air = Aircraft(aircraft_params)
    # air.write_mesh("panel", "vae_output.x", 0)
    # print("已生成VAE输出网格！")