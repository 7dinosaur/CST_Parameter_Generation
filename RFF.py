import numpy as np

# --------------------------
# RFF 映射：10维 → 20维
# --------------------------
np.random.seed(42)  # 固定种子，保证映射永远一致
n_low = 10          # 你真正优化的变量
n_high = 20         # 映射后的高维变量

# 随机傅里叶特征参数（固定）
W = np.random.randn(n_low, n_high) * 1.2   # 投影矩阵
b = np.random.rand(n_high) * 2 * np.pi     # 相位偏移

def lift(x):
    """低维 -> 高维（确定性、非线性、分布均匀）"""
    return np.sin(x @ W + b)

def test():
    x = np.random.rand(n_low)
    y = lift(x)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("y:\n", y)

if __name__ == "__main__":
    test()