import os
import sys
import subprocess
from monty.serialization import loadfn


def run_command(command):
    """
    在程序外部运行linux命令，并实时输出
    :param command: 需要执行的命令，输入为一个list，每个元素为一个str。例如：["mpirun", "-n", "2", "abacus"]
    :return: None
    """
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed! {command}")
        sys.exit(result.returncode)


def main():
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 main.py $(your configuration filename)")
        print("Error: Please provide the configuration file")
        sys.exit(1)

    cwd = os.getcwd()

    # 记录配置文件名，加载配置文件
    config_file = sys.argv[1]
    config = loadfn(config_file)

    # 获取该程序文件所在目录
    main_dir = os.path.dirname(os.path.abspath(__file__))

    # 读取是否要进行能量差分计算。默认不进行
    run_efd: int = config["run_efd"] if "run_efd" in config else 0

    # 读取要使用的软件，ABACUS或LAMMPS
    calculator = config["calculator"]

    # 读取设定的openmpi并行线程数，mpirun并行进程数。默认OMP_NUM_THREADS=1，mpirun -n 1
    OMP_NUM_THREADS: int = config["OMP_NUM_THREADS"] if "OMP_NUM_THREADS" in config else 1
    MPIRUN_NUM_PROC: int = config["MPIRUN_NUM_PROC"] if "MPIRUN_NUM_PROC" in config else 1

    # 执行make
    run_command(f"python3 {main_dir}/make.py {config_file}")

    # 定位work文件夹
    work_dir = "./work"
    # 遍历结构文件夹
    for stru in os.listdir(work_dir):
        stru_path = os.path.join(work_dir, stru)
        # 跳过可能存在的非文件夹
        if not os.path.isdir(stru_path):
            continue
        # 遍历task任务文件夹
        for task in os.listdir(stru_path):
            # 限定搜索task文件夹
            if not task.startswith("task."):
                continue
            task_path = os.path.join(stru_path, task)
            # 进入task任务文件夹
            os.chdir(task_path)
            # 提交计算任务
            if calculator == "abacus":
                run_command(f"OMP_NUM_THREADS={OMP_NUM_THREADS} mpirun -n {MPIRUN_NUM_PROC} abacus")
            elif calculator == "lammps":
                run_command(f"mpirun -n {MPIRUN_NUM_PROC} lmp -i in.lammps")
            else:
                raise ValueError("Only support ABACUS or LAMMPS calculator")
            # 返回工作目录
            os.chdir(cwd)

    # 如果config中有设置，进行能量差分计算操作
    if run_efd:
        # 执行EFD_make
        run_command(f"python3 {main_dir}/EFD_make.py {config_file}")

        # 遍历结构文件夹
        for stru in os.listdir(work_dir):
            stru_path = os.path.join(work_dir, stru)
            # 跳过可能存在的非文件夹
            if not os.path.isdir(stru_path):
                continue
            # 遍历task任务文件夹
            for task in os.listdir(stru_path):
                # 限定搜索task文件夹
                if not task.startswith("task."):
                    continue
                task_path = os.path.join(stru_path, task)
                efd_path = os.path.join(task_path, "EFD")
                # 遍历微小变胞文件夹
                for diff in os.listdir(efd_path):
                    diff_path = os.path.join(efd_path, diff)
                    if os.path.isdir(diff_path):
                        # 进入微小变胞文件夹，提交计算任务
                        os.chdir(diff_path)
                        if calculator == "abacus":
                            run_command(f"OMP_NUM_THREADS={OMP_NUM_THREADS} mpirun -n {MPIRUN_NUM_PROC} abacus")
                        elif calculator == "lammps":
                            run_command(f"mpirun -n {MPIRUN_NUM_PROC} lmp -i in.lammps")
                        else:
                            raise ValueError("Only support ABACUS or LAMMPS calculator")
                        # 返回工作目录
                        os.chdir(cwd)

            # 定位结构文件目录下的EFD_task文件夹
            efd_task_path = os.path.join(stru_path, "EFD_task")
            # 遍历微小变胞文件夹
            for diff in os.listdir(efd_task_path):
                diff_path = os.path.join(efd_task_path, diff)
                if os.path.isdir(diff_path):
                    # 进入微小变胞文件夹，提交计算任务
                    os.chdir(diff_path)
                    if calculator == "abacus":
                        run_command(f"OMP_NUM_THREADS={OMP_NUM_THREADS} mpirun -n {MPIRUN_NUM_PROC} abacus")
                    elif calculator == "lammps":
                        run_command(f"mpirun -n {MPIRUN_NUM_PROC} lmp -i in.lammps")
                    else:
                        raise ValueError("Only support ABACUS or LAMMPS calculator")
                    # 返回工作目录
                    os.chdir(cwd)

        # 执行EFD_post
        run_command(f"python3 {main_dir}/EFD_post.py {config_file}")

    # 执行post
    run_command(f"python3 {main_dir}/post.py {config_file}")


if __name__ == '__main__':
    main()
