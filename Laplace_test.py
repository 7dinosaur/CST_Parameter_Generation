from matplotlib.pylab import laplace

from aircraft_gene import Aircraft
import numpy as np
from gene_aircraft_MPI import generate_one_candidate
import pandas as pd

if __name__ == "__main__":
    base_para = pd.read_csv("smooth_test.csv").to_numpy()
    base_air = Aircraft(base_para)
    base_laplace = base_air.Laplace()
    print("基准样本的几何光顺性指标（Laplace） =", base_laplace)
    generated_para, pas, laplace = generate_one_candidate(base_para, base_laplace)
    new_air = Aircraft(generated_para)
    new_air.write_mesh("simple", "test.x")
    print(f"生成样本的几何光顺性指标（Laplace） = {laplace}, 载客量 = {pas}")