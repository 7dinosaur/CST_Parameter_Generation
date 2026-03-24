from concurrent.futures import ProcessPoolExecutor
from cal_Lift import cal_Lift
import time

def run_task(interp_id):
    print(f"开始运行任务 {interp_id}")
    base_path = f"MPI_FABOOM\\FABOOM_0{interp_id}"
    result = cal_Lift(base_path)

if __name__ == "__main__":
    task_list = [1, 2, 3, 4]  # 任务列表
    start = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_task, task_list))
    end = time.time()
    print(f"所有任务完成，耗时 {end - start:.2f} 秒")