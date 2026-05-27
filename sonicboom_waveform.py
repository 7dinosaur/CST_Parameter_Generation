"""
基于等效截面积分布计算声爆 F 函数与近场波形

算法严格遵循 FABOOM.f90 的 diferrential → F_function 流程:

1. 总等效截面积 S(x) → 二阶中心差分 → S''(x)
2. Whitham 积分 → F(y) = 1/(2π) ∫ S''(ξ)/√(y-ξ) dξ
3. 尾流延伸 (wake extension)
4. 近场压力波形: ΔP/P₀ = γ·M²/√(2·B·r) · F(y)

用法:
    python sonicboom_waveform.py <area.dat> [--mach M] [--HL HL] [--p0 P0]

输入文件格式 (与 FABOOM sonicboom area.dat 相同):
    N
    x(1)  S(1)
    x(2)  S(2)
    ...
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


# =============================================================================
# 二阶差分 — 对应 FABOOM subroutine diferrential
# =============================================================================
def compute_second_derivative(x, S):
    """中心差分计算 S''(x), 严格匹配 FABOOM subroutine diferrential

    FABOOM 索引 (1-based):
      NUM = read_count + 1
      x(1)=x(2)-x(3), y(1)=0  — 人工前导零点
      x(2:NUM), y(2:NUM)      — 实际数据
      dy2(i) = (y(i+1)+y(i-1)-2·y(i)) / (x(i+1)-x(i))²   i=2..NUM-1
    """
    N_in = len(x)
    N = N_in + 1
    x_aug = np.empty(N)
    S_aug = np.empty(N)
    x_aug[0] = x[0] - (x[1] - x[0])  # 人工前导点
    S_aug[0] = 0.0
    x_aug[1:] = x
    S_aug[1:] = S

    Spp = np.zeros(N)
    for i in range(1, N - 1):   # FABOOM: i=2..NUM-1
        dx = x_aug[i + 1] - x_aug[i]
        # Spp[i] = (S[i+1] + S[i-1] - 2·S[i]) / (x[i+1]-x[i])²
        Spp[i] = (S_aug[i + 1] + S_aug[i - 1] - 2.0 * S_aug[i]) / (dx * dx)
    Spp[0] = 0.0
    Spp[N - 1] = 0.0

    return x_aug, Spp


# =============================================================================
# F 函数 — 对应 FABOOM subroutine F_function
# =============================================================================
def compute_F_function(x, Spp, Nw=50, Nc=10):
    """Whitham F-函数积分, 含尾流延伸

    严格遵循 FABOOM subroutine F_function:
      - 对每个 y(j)=x(j), 重新积分 ∫₀^{x(j-1)} S''(ξ)/√(y(j)-ξ) dξ
      - 末点用线性外推: F(j) = 2·F(j-1) - F(j-2)  (避免 ξ→y(j) 奇点)
      - 尾流: F_w(x) = -1/π/√(x-y_N) · ∫₀^{y_N} F(ξ)·√(y_N-ξ)/(x-ξ) dξ
    """
    N = len(x)
    y = x.copy()

    # ---- 主 F-函数 ----
    F_arr = np.zeros(N)     # 临时工作数组, 每次 j 循环重新填充
    F_hanshu = np.zeros(N)  # 最终输出的 F(y)/2π

    for j in range(N):
        yj = y[j]

        # F(1) = 0
        F_arr[0] = 0.0

        if N >= 2:
            F_arr[1] = Spp[1] * (x[1] - x[0]) / np.sqrt(x[1] - x[0])

        if j > 1:
            # 对当前 yj 重新积分: F(i) = ∫₀^{x(i)} S''(ξ)/√(yj-ξ) dξ, i=2..j-1
            for i in range(1, j):  # i=1 → 更新 F_arr[1]; i=2..j-1 → 更新 F_arr[2..j-1]
                if i == 0:
                    continue
                dx = x[i] - x[i - 1]
                denom_i = yj - x[i]
                denom_im1 = yj - x[i - 1]
                if denom_i <= 0 or denom_im1 <= 0:
                    continue
                F_arr[i] = F_arr[i - 1] + 0.5 * dx * (
                    Spp[i] / np.sqrt(denom_i)
                    + Spp[i - 1] / np.sqrt(denom_im1)
                )
            # 外推到 yj=x(j): F(j) = 2·F(j-1) - F(j-2)
            F_arr[j] = 2.0 * F_arr[j - 1] - F_arr[j - 2]

        F_hanshu[j] = F_arr[j] / (2.0 * np.pi)

    # ---- 尾流延伸 ----
    xw = np.linspace(y[-1] + 0.8 * y[-1] / Nw,
                     y[-1] + 0.8 * y[-1],
                     Nw)

    Fw = np.zeros(Nw)
    yN = y[-1]
    for iw in range(Nw):
        s = 0.0
        for j in range(N - 1):
            # [y(j), y(j+1)] 细分为 Nc 段
            xc = np.linspace(y[j], y[j + 1], Nc)
            fc = np.linspace(F_hanshu[j], F_hanshu[j + 1], Nc)
            for kk in range(Nc - 1):
                dxc = xc[kk + 1] - xc[kk]
                s += 0.5 * dxc * (
                    fc[kk] * np.sqrt(max(yN - xc[kk], 0)) / (xw[iw] - xc[kk])
                    + fc[kk + 1] * np.sqrt(max(yN - xc[kk + 1], 0)) / (xw[iw] - xc[kk + 1])
                )
        Fw[iw] = -s / (np.pi * np.sqrt(xw[iw] - yN))

    y_full = np.concatenate([y, xw])
    F_full = np.concatenate([F_hanshu, Fw])

    return y_full, F_full


# =============================================================================
# 近场压力波形 — 对应 FABOOM F_function 末尾 delta_P 部分
# =============================================================================
def compute_nearfield(y, F, M, HL, L, p0=101325.0, gamma=1.4):
    """从 F-函数计算近场过压分布

    Parameters
    ----------
    y, F : F-函数
    M : 马赫数
    HL : H/L, 传播距离比 (r = L * HL)
    L : 机身长度 (m)
    p0 : 环境静压 (Pa), 默认海平面
    gamma : 比热比, 默认 1.4

    Returns
    -------
    x_nf : 非线性修正后的纵轴坐标
    dp_p0 : ΔP/P₀ 过压比
    """
    B = np.sqrt(M * M - 1.0)  # Prandtl-Glauert 因子
    r = L * HL                # 传播距离

    # 非线性因子: k = 1/√2 · (γ+1) · M⁴ · B^(-3/2)
    k = 1.0 / np.sqrt(2.0) * (gamma + 1.0) * M**4 * B**(-1.5)

    # ΔP/P₀ = γ·M² / √(2·B·r) · F(y)
    amp = gamma * M * M / np.sqrt(2.0 * B * r)
    dp_p0 = amp * F

    # x 非线性前移/后移: x' = y - k · F(y) · √r
    x_nf = y - k * F * np.sqrt(r)

    return x_nf, dp_p0


# =============================================================================
# 高层封装
# =============================================================================
def sonicboom_from_area(area_file, M, HL=3.0, L=72.0, p0=101325.0, gamma=1.4,
                        Nw=50, Nc=10):
    """从总等效截面积文件计算完整声爆近场波形

    Parameters
    ----------
    area_file : str  面积文件, 支持两种格式:
        (a) sonicboom area.dat: 第一行 N, 随后 N 行 x S
        (b) area_distribution2 输出: 首行 VARIABLES=..., 随后 x S
    M : float  马赫数
    HL : float  H/L 传播距离比 (r = L·HL), FABOOM 默认 3.0
    L : float  机身长度 (m), 默认 72
    p0 : float  环境静压 (Pa), 默认海平面 101325
    gamma : float  比热比, 默认 1.4
    """
    # 自动检测文件格式
    with open(area_file, 'r') as f:
        first = f.readline().strip()
    # 尝试解析首行: 若是数字则为 sonicboom 格式 (N), 否则为 VARIABLES 头
    try:
        N = int(first.split()[0]) if first else 0
        data = np.loadtxt(area_file, skiprows=1)
        data = data[:N]
    except (ValueError, IndexError):
        data = np.loadtxt(area_file, skiprows=1)

    x = data[:, 0]
    S = data[:, 1]

    # 二阶导数
    x_aug, Spp = compute_second_derivative(x, S)

    # F-函数 (含尾流)
    y, F = compute_F_function(x_aug, Spp, Nw=Nw, Nc=Nc)

    # 近场波形
    B = np.sqrt(M * M - 1.0)
    r = L * HL
    k = 1.0 / np.sqrt(2.0) * (gamma + 1.0) * M**4 * B**(-1.5)
    amp = gamma * M * M / np.sqrt(2.0 * B * r)
    dp_p0 = amp * F
    x_nf = y - k * F * np.sqrt(r)

    return {
        'x': x, 'S': S,
        'x_aug': x_aug, 'Spp': Spp,
        'y': y, 'F': F,
        'x_nf': x_nf, 'dp_p0': dp_p0,
        'M': M, 'HL': HL, 'L': L, 'p0': p0, 'gamma': gamma,
        'B': B, 'r': r, 'k': k, 'amp': amp,
    }


# =============================================================================
# 绘图
# =============================================================================
def plot_results(res, save_prefix=None):
    """绘制 S(x), S''(x), F(y), ΔP/P₀ 四联图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) 等效截面积
    ax = axes[0, 0]
    ax.plot(res['x'], res['S'], 'b-', lw=1)
    ax.set_xlabel('x'); ax.set_ylabel('S(x)')
    ax.set_title('Total Equivalent Area')
    ax.grid(True, alpha=0.3)

    # (0,1) 二阶导数
    ax = axes[0, 1]
    ax.plot(res['x_aug'], res['Spp'], 'r-', lw=1)
    ax.set_xlabel('x'); ax.set_ylabel("S''(x)")
    ax.set_title('Second Derivative')
    ax.grid(True, alpha=0.3)

    # (1,0) F-函数
    ax = axes[1, 0]
    ax.plot(res['y'], res['F'], 'g-', lw=1)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('y'); ax.set_ylabel('F(y)')
    ax.set_title('Whitham F-function')
    ax.grid(True, alpha=0.3)

    # (1,1) 近场波形
    ax = axes[1, 1]
    ax.plot(res['x_nf'], res['dp_p0'], 'm-', lw=1)
    ax.axhline(y=0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('x (corrected)'); ax.set_ylabel(r'$\Delta P / P_0$')
    ax.set_title(f"Near-field Signature (M={res['M']:.2f}, r/L={res['HL']:.1f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f'{save_prefix}_waveform.png', dpi=120)
        print(f'Plot saved: {save_prefix}_waveform.png')
    return fig


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基于等效截面积分布计算声爆 F 函数与近场波形 (FABOOM 算法)')
    parser.add_argument('area_file', help='总等效截面积文件 (sonicboom area.dat 格式)')
    parser.add_argument('--mach', '-M', type=float, default=1.8,
                        help='马赫数 (默认 1.8)')
    parser.add_argument('--HL', type=float, default=3.0,
                        help='H/L 传播距离比 (默认 3.0, FABOOM 默认)')
    parser.add_argument('--L', type=float, default=72.0,
                        help='机身长度 m (默认 72)')
    parser.add_argument('--p0', type=float, default=101325.0,
                        help='环境静压 Pa (默认 101325 海平面)')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='比热比 (默认 1.4)')
    parser.add_argument('--Nw', type=int, default=50,
                        help='尾流延伸点数 (默认 50)')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='保存图片前缀')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出近场波形到文件')
    args = parser.parse_args()

    print(f'Area file: {args.area_file}')
    print(f'Params: M={args.mach}, HL={args.HL}, L={args.L}, p0={args.p0}, gamma={args.gamma}')

    res = sonicboom_from_area(
        args.area_file, args.mach, HL=args.HL, L=args.L, p0=args.p0,
        gamma=args.gamma, Nw=args.Nw)

    print(f'Area data points: {len(res["x"])}')
    print(f'B = {res["B"]:.4f}, r = {res["r"]:.1f} m')
    print(f'k = {res["k"]:.6f}, amp = {res["amp"]:.6f}')
    print(f'F-function range: [{res["F"].min():.4f}, {res["F"].max():.4f}]')
    dp = res['dp_p0']
    print(f'Nearfield dP/P0 range: [{dp.min():.4f}, {dp.max():.4f}]')

    if args.output:
        out = np.column_stack([res['x_nf'], res['dp_p0']])
        header = 'variables=x-Bh,<greek>D</greek>p/p<sub>0</sub>\n-1 0'
        np.savetxt(args.output, out, header=header, comments='', fmt='%.8f')
        print(f'Nearfield waveform saved to: {args.output}')

    plot_results(res, save_prefix=args.save)
    plt.show()
