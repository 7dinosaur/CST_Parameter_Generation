import os
import shutil
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

# ====================== 你的配置（直接改这里）======================
ORIGINAL_DIR = r"FABOOM_test\bboom"  # 原始文件夹
PARALLEL_NUM = 5                                 # 并行几个任务
BASE_WORK_DIR = r"FABOOM_parallel_tasks"          # 自动生成的工作目录
# =================================================================

def run_single_task(task_id):
    """单个任务：复制文件夹 → 运行exe → 计时"""
    task_dir = os.path.join(BASE_WORK_DIR, f"task_{task_id}")
    
    # 1. 复制整个原始文件夹（完全复制，不修改任何内容）
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    shutil.copytree(ORIGINAL_DIR, task_dir)

    exe_path = os.path.join(task_dir, "bBoom.exe")
    result_path = os.path.join(task_dir, "OUTDATA", "Burgers_NOISE.DAT")

    print(f"[任务 {task_id}] 开始计算 → {task_dir}")
    start = time.time()

    try:
        # 2. 运行exe（独立目录，完全不冲突）
        result = subprocess.run(
            [exe_path],
            cwd=task_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 3. 读取结果
        val = float([*open(result_path)][3].split()[2])
        cost = round(time.time() - start, 2)
        print(f"[任务 {task_id}] 完成！结果={val:.4f} 耗时={cost}s")
        return task_id, val, cost

    except Exception as e:
        cost = round(time.time() - start, 2)
        print(f"[任务 {task_id}] 失败！错误={str(e)} 耗时={cost}s")
        return task_id, -1, cost

def run_parallel_test():
    """并行测试主函数"""
    print("="*60)
    print(f"开始并行测试 | 原始目录：{ORIGINAL_DIR} | 并行数：{PARALLEL_NUM}")
    print("="*60)

    os.makedirs(BASE_WORK_DIR, exist_ok=True)
    total_start = time.time()

    # 多线程并行启动
    with ThreadPoolExecutor(max_workers=PARALLEL_NUM) as executor:
        futures = [executor.submit(run_single_task, i) for i in range(PARALLEL_NUM)]
        results = [f.result() for f in futures]

    total_cost = round(time.time() - total_start, 2)

    # 输出最终统计
    print("\n" + "="*60)
    print(f"【全部完成】总耗时：{total_cost} 秒")
    for task_id, val, cost in results:
        print(f"任务 {task_id} | 结果：{val:.4f} | 耗时：{cost}s")
    print("="*60)

if __name__ == "__main__":
    run_parallel_test()