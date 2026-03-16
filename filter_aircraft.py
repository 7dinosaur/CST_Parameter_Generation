import numpy as np
import pandas as pd
from aircraft_gene import Aircraft

if __name__ == "__main__":
    para_list = pd.read_csv("qualified_solutions.csv").to_numpy()
    mask = (para_list[:, 1] < 2000000.0) & (para_list[:, 2] > 120) & (para_list[:, 1] > 1250000.0)
    geo_list = para_list[mask]
    print(geo_list.shape)
    new_solutions = []
    for solution in geo_list:
        geo = solution[3:].reshape([-1, 24])
        test_aircraft = Aircraft(geo)
        test_aircraft.write_mesh("panel", "check.x")
        print(f"升力为：{solution[1]}, 载客量为： {solution[2]}")
        if_pass = input("是否合格：")

        if if_pass == "1":
            print("合格")
            new_solutions.append(solution)
        else:
            print("不合格")
    new_solutions = np.array(new_solutions)
    csv_solutions = pd.DataFrame(new_solutions)
    csv_solutions.to_csv("filter_solutions_v2.csv", index=False)