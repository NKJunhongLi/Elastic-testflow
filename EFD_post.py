import os
import sys

import numpy as np
from monty.serialization import loadfn, dumpfn
from glob import glob
from Conf import Conf
from calculator.ABACUS import parse_abacus_log


def __EFD_post__():
    """
        在执行完EFD_make并完成全部自洽计算任务后，解析输出文件，完成能量的中心差分计算，得到应力stress。\n
        每个task对应结构的应力张量stress.json文件存储在EFD文件夹内。
    """
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 EFD_post.py $(your configuration filename)")
        print("Error: Please provide the configuration file")
        sys.exit(1)

    # 加载配置文件，使用monty.serialization.loadfn()以获得更高兼容性
    config = loadfn(sys.argv[1])

    cwd = os.getcwd()
    # make步骤的工作文件夹为work
    work_path = os.path.join(cwd, "work")
    # 检查是否进行了make操作。如果没有，报error并退出程序
    if not os.path.exists(work_path):
        print("Error: Work path not found! Please do make first!")
        sys.exit(1)

    if config["calculator"] == "abacus":
        # make前提供的初始结构文件路径，存在一个list里
        origin_structures = [os.path.join(cwd, file) for file in config["stru_files"]]

        # 这个变量用于访问config字典中"relax_log_dir"对应的list元素
        # strufile_index: int = 0

        for origin_strufile in origin_structures:
            # 记录结构文件的文件名，用于接下来进入"work"目录下的结构操作文件夹
            stru_filename = os.path.basename(origin_strufile)

            # 使用glob()函数找到24个task文件夹并存到一个list里
            task_dirs = sorted(glob(os.path.join(work_path, stru_filename, "task.*")))

            # 做能量差分计算的文件夹在task文件夹下，名为EFD。将它们全部存到一个list里
            efd_dirs = [os.path.join(task, "EFD") for task in task_dirs]
            # 把初始结构能量差分的EFD_task文件夹也添加到list里
            efd_dirs.append(os.path.join(work_path, stru_filename, "EFD_task"))

            for efd in efd_dirs:
                # 检查路径是否存在，判断有没有做EFD_make。如果没有，报error并退出程序
                if not os.path.exists(efd):
                    print(f"Error: EFD path {efd} not found! Please do EFD_make first!")
                    sys.exit(1)

                # 进入EFD文件夹
                os.chdir(efd)

                # 找到并读取结构文件STRU
                stru_file = os.path.join(efd, "STRU")
                '''
                # 如果是初始结构的EFD_task，结构文件即是配置文件给出的STRU
                if os.path.basename(efd) == "EFD_task":
                    stru_file = origin_strufile
                '''
                conf = Conf.from_abacus(stru_file)
                # 计算体积，用于后面代入公式计算应力
                volume = conf.get_volume()

                '''
                # 定位task文件夹的relax任务的输出log，task文件夹是EFD文件夹的上一级目录
                relax_log = os.path.join(efd, os.path.pardir, "OUT.ABACUS/running_relax.log")
                # 如果task的log文件不存在，报error并退出程序
                if not os.path.exists(relax_log):
                    print("Error: Output files for relax tasks not found! Please run relax tasks first!")
                    sys.exit(1)
                # 解析日志文件并获取总能
                relax_E0 = parse_log(relax_log)["final_etot"]
                '''

                stress_dict = {}  # 应力分量字典，存储计算得到的应力分量各自对应的数值
                # 对6个分量，各自对应2个文件夹内容，逐个进行操作
                for index in ["xx", "yy", "zz", "yz", "xz", "xy"]:
                    # 定位scf计算的输出目录，p结尾变量名对应施加+小应变的，n结尾变量名对应施加-小应变的。
                    scf_output_p = os.path.join(efd, f"{index}+", "OUT.ABACUS")
                    scf_output_n = os.path.join(efd, f"{index}-", "OUT.ABACUS")
                    # 检查scf输出文件是否存在。如果不存在，报error并退出程序
                    if (
                            not os.path.exists(scf_output_p)
                            or not os.path.isdir(scf_output_p)
                            or not os.path.exists(scf_output_n)
                            or not os.path.isdir(scf_output_n)
                    ):
                        print("Error: Output path for EFD tasks not found! Please run EFD tasks first!")
                        sys.exit(1)

                    # 定位自洽计算的log文件
                    scf_log_p = os.path.join(scf_output_p, "running_scf.log")
                    scf_log_n = os.path.join(scf_output_n, "running_scf.log")
                    # 读取log文件中的总能量数据
                    scf_Eij_p = parse_abacus_log(scf_log_p)["final_etot"]
                    scf_Eij_n = parse_abacus_log(scf_log_n)["final_etot"]

                    # 读取strain.json获得应变张量，并提取非零的数值存储为分量值。根据公式规范，这里应当取正值
                    strain_tensor = loadfn(os.path.join(efd, f"{index}+", "strain.json"))
                    # 用strain_tensor != 0创建布尔掩码，对应非零元素的位置
                    nonzero_element = strain_tensor[strain_tensor != 0]
                    # 应力张量是对称矩阵，取第1个非零值即可。用item()函数提取值返回float类型
                    strain_value = nonzero_element[0].item()

                    # 测试：添加对角项修正
                    # if index == "xx" or index == "yy" or index == "zz":
                    #     strain_value /= 1 + strain_value

                    # 中心差分的应力计算公式：stress_value = (Eij_n - Eij_p) / (2*strain_value * volume)
                    stress_value = (scf_Eij_n - scf_Eij_p) / (2 * strain_value) / volume
                    # 将数值结果存到字典里
                    stress_dict[index] = stress_value

                # 构造一个3*3的numpy矩阵用于存储应力张量结果
                stress_tensor = np.zeros((3, 3))
                # 根据stress_dict字典里存储的数据，逐个给stress_tensor的元素赋值
                stress_tensor[(0, 0)] = stress_dict["xx"]
                stress_tensor[(1, 1)] = stress_dict["yy"]
                stress_tensor[(2, 2)] = stress_dict["zz"]
                stress_tensor[(1, 2)] = stress_dict["yz"]
                stress_tensor[(2, 1)] = stress_dict["yz"]
                stress_tensor[(0, 2)] = stress_dict["xz"]
                stress_tensor[(2, 0)] = stress_dict["xz"]
                stress_tensor[(0, 1)] = stress_dict["xy"]
                stress_tensor[(1, 0)] = stress_dict["xy"]

                # 单位(eV/Angtrom^3)到(KBar)的转换系数
                stress_tensor *= 1602.1757722389546

                # 将应力张量计算结果输出
                dumpfn(stress_tensor, "stress.json", indent=4)

                # 回到工作目录
                os.chdir(work_path)

            # strufile_index += 1

    elif config["calculator"] == "lammps":
        return

    else:
        raise ValueError("Only support ABACUS or LAMMPS calculator")

    return


if __name__ == "__main__":
    __EFD_post__()
