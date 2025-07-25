import sys
import os
from monty.serialization import loadfn
from calculator.ABACUS import parse_abacus_log


def benchmark(work_path):
    """
    根据给出的路径定位工作文件夹work，遍历所有计算任务，计算每个任务的能量差分stress结果和abacus解析stress结果的相对误差。\n
    在屏幕上输出矩阵所有分量的相对误差，并在最后输出所有任务中最大相对误差和最大绝对误差数值
    :param work_path:
    :return: 无返回值，仅在屏幕上打印信息
    """
    # 初始化存储所有误差的list
    all_relative_errors = []
    all_absolute_errors = []
    # 遍历work目录下所有结构文件夹
    for stru_file in os.listdir(work_path):
        stru_path = os.path.join(work_path, stru_file)
        # 跳过非文件夹
        if not os.path.isdir(stru_path):
            continue
        # 遍历结构文件夹下的所有task文件夹
        for task_file in os.listdir(stru_path):
            task_path = os.path.join(stru_path, task_file)
            # 跳过非文件夹
            if not os.path.isdir(task_path):
                continue
            abacus_log = os.path.join(task_path, "OUT.ABACUS", "running_relax.log")
            efd_stress_json = os.path.join(task_path, "EFD", "stress.json")
            # 检查log文件和应力文件是否存在
            if not os.path.exists(abacus_log):
                print(f"警告: ABACUS日志文件不存在 - {abacus_log}")
                continue
            if not os.path.exists(efd_stress_json):
                print(f"警告: EFD应力文件不存在 - {efd_stress_json}")
                continue

            abacus_stress = parse_abacus_log(abacus_log)["stress"]
            efd_stress = loadfn(efd_stress_json)

            # 打印当前任务信息
            print(f"\n{'=' * 50}")
            print(f"Structure: {stru_file}, task: {task_file}")
            print("-" * 50)

            # 计算并打印每个分量的相对误差
            print("\nRelative Error (%):")
            for i in range(3):
                for j in range(3):
                    # 计算绝对误差
                    abs_error = abs(efd_stress[i, j] - abacus_stress[i, j])
                    all_absolute_errors.append(abs_error)

                    # 计算相对误差（避免除以零）
                    if abs(abacus_stress[i, j]) > 1e-7:  # 设置阈值避免除零
                        rel_error = 100 * abs_error / abs(abacus_stress[i, j])
                        all_relative_errors.append(rel_error)
                        print(f"  ({i},{j}): {rel_error:>10.2f}%")
                    else:
                        if abs_error < 1e-4:
                            rel_error = 0.0
                            print(f"  ({i},{j}): {rel_error:>10.2f}%")
                        else:
                            print(f"  ({i},{j}): {'N/A':>15}")

            print(f"{'=' * 50}")

    # 计算并输出统计结果
    print("\n\n" + "=" * 60)
    print("Final statistic results:")
    print("=" * 60)

    print(f"最大相对误差: {max(all_relative_errors):.2f}%")
    print(f"最大绝对误差: {max(all_absolute_errors):.2f} KBAR")

    print("=" * 60)


if __name__ == "__main__":
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py $(the work directory path)")
        print("Error: Please provide the work path")
        sys.exit(1)

    work_path = os.path.join(sys.argv[1])
    benchmark(work_path)
