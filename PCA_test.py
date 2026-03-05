import numpy as np
from sklearn.preprocessing import StandardScaler

class CST_PCA:
    """
    针对10×10 CST参数的PCA降维与还原类
    支持：降维到指定低维→优化低维参数→还原回10×10 CST参数
    """
    def __init__(self, n_components: int = 5):
        """
        参数:
            n_components: 降维后的低维设计变量数（如5）
        """
        self.n_components = n_components
        self.scaler = StandardScaler()  # 标准化器
        self.mean = None  # 原始数据的均值（去中心化用）
        self.projection_matrix = None  # 降维投影矩阵（特征向量）
        self.explained_variance_ratio = None  # 方差贡献率
    
    def fit(self, cst_data: np.ndarray):
        """
        训练PCA模型（基于多个CST样本）
        参数:
            cst_data: 形状为 (样本数, 10, 10) 或 (样本数, 100) 的CST数据
        """
        # 统一数据形状为 (样本数, 100)
        if len(cst_data.shape) == 3:
            cst_data = cst_data.reshape(cst_data.shape[0], -1)
        
        # 步骤1：标准化（消除量纲影响，关键）
        cst_scaled = self.scaler.fit_transform(cst_data)
        self.mean = self.scaler.mean_  # 保存均值（还原时用）
        self.scale = self.scaler.scale_  # 保存标准差（还原时用）
        
        # 步骤2：计算协方差矩阵
        cov_matrix = np.cov(cst_scaled, rowvar=False)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # 确保实对称
        
        # 步骤3：特征值分解并排序
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # 步骤4：保存投影矩阵（前n_components个特征向量）
        self.projection_matrix = sorted_eigenvectors[:, :self.n_components]
        
        # 计算方差贡献率
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio = sorted_eigenvalues[:self.n_components] / total_variance
        
        print(f"PCA训练完成！累计方差贡献率：{np.sum(self.explained_variance_ratio):.4f}")
    
    def transform(self, cst_data: np.ndarray) -> np.ndarray:
        """
        降维：将10×10 CST参数转换为低维设计变量（如5维）
        参数:
            cst_data: 形状为 (样本数, 10, 10) 或 (样本数, 100) 的CST数据
        返回:
            low_dim_data: 形状为 (样本数, n_components) 的低维变量
        """
        if self.projection_matrix is None:
            raise ValueError("请先调用fit()训练PCA模型！")
        
        # 统一形状并标准化
        if len(cst_data.shape) == 3:
            cst_data = cst_data.reshape(cst_data.shape[0], -1).astype(np.float64)
        cst_scaled = self.scaler.transform(cst_data)
        
        # 降维：投影到主成分空间
        low_dim_data = np.dot(cst_scaled, self.projection_matrix).astype(np.float64)
        return low_dim_data
    
    def inverse_transform(self, low_dim_data: np.ndarray) -> np.ndarray:
        """
        解码还原：将优化后的低维变量还原为10×10 CST参数
        参数:
            low_dim_data: 形状为 (样本数, n_components) 的低维变量
        返回:
            restored_cst: 形状为 (样本数, 10, 10) 的还原后CST参数
        """
        if self.projection_matrix is None:
            raise ValueError("请先调用fit()训练PCA模型！")
        
        # 步骤1：从低维空间投影回标准化后的高维空间
        cst_scaled_restored = np.dot(low_dim_data, self.projection_matrix.T)
        
        # 步骤2：逆标准化（还原原始量纲）
        cst_restored = self.scaler.inverse_transform(cst_scaled_restored)
        
        # 步骤3：重塑为10×10的CST结构
        restored_cst = cst_restored.reshape(low_dim_data.shape[0], 10, 10)
        return restored_cst

# ------------------- 完整测试流程（降维→优化→还原） -------------------
if __name__ == "__main__":
    # 1. 生成模拟数据：50个样本，每个样本10×10 CST参数（模拟工程样本）
    np.random.seed(42)
    cst_samples = np.random.randn(50, 10, 10)  # 50个样本，10×10 CST
    
    # 2. 初始化并训练PCA模型（降维到5维）
    pca = CST_PCA(n_components=5)
    pca.fit(cst_samples)
    
    # 3. 取一个测试样本（比如第0个样本），降维成5个设计变量
    test_cst = cst_samples[0:1, :, :]  # 形状(1,10,10)
    low_dim_vars = pca.transform(test_cst)
    print(f"低维设计变量（5维）：\n{np.round(low_dim_vars, 4)}")
    
    # 4. 模拟优化：对低维变量做调整（比如+0.5，代表优化后的结果）
    optimized_low_dim = low_dim_vars + 0.5
    print(f"\n优化后的低维变量：\n{np.round(optimized_low_dim, 4)}")
    
    # 5. 解码还原：将优化后的5维变量还原为10×10 CST参数
    restored_cst = pca.inverse_transform(optimized_low_dim)
    print(f"\n还原后的CST参数形状：{restored_cst.shape}")
    print(f"还原后第1个剖面的前5个CST参数：\n{np.round(restored_cst[0, 0, :5], 4)}")
    
    # 6. 验证信息保留程度（可选）
    # 计算原始样本与还原样本的均方误差（越小说明还原效果越好）
    mse = np.mean((test_cst - pca.inverse_transform(low_dim_vars))**2)
    print(f"\n原始样本→降维→还原的均方误差：{mse:.6f}")