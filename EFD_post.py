import os
import sys
import numpy as np

from monty.serialization import loadfn, dumpfn
from glob import glob
from Conf import Conf
from calculator.ABACUS import parse_abacus_log
from calculator.LAMMPS import parse_lammps_log
from make import clean_matrix

# ABACUS和LAMMPS软件使用不同的内部定义常数数值，因此单位转换系数在第7位有效数字后有差距
ABACUS_UnitTrans = 1602.1757722389546  # 从(eV/Angstrom^3)到KBar
LAMMPS_UnitTrans = 1602176.634  # 从(eV/Angstrom^3)到Bar


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

    if config["calculator"] not in ["abacus", "lammps"]:
        raise ValueError("Only support ABACUS or LAMMPS calculator")

    cwd = os.getcwd()
    # make步骤的工作文件夹为work
    work_path = os.path.join(cwd, "work")
    # 检查是否进行了make操作。如果没有，报error并退出程序
    if not os.path.exists(work_path):
        print("Error: Work path not found! Please do make first!")
        sys.exit(1)

    # make前提供的初始结构文件路径，存在一个list里
    origin_structures = [os.path.join(cwd, file) for file in config["stru_files"]]

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

            if config["calculator"] == "abacus":
                # 找到并读取结构文件STRU
                stru_file = os.path.join(efd, "STRU")
                conf = Conf.from_abacus(stru_file)

            elif config["calculator"] == "lammps":
                stru_file = os.path.join(efd, "conf.lmp")
                conf = Conf.from_lammps(stru_file, config["interaction"]["type_map"])

            else:
                raise ValueError("Only support ABACUS or LAMMPS calculator")

            # 计算体积，用于后面代入公式计算应力
            volume = conf.get_volume()

            stress_dict = {}  # 应力分量字典，存储计算得到的应力分量各自对应的数值

            # 对6个分量，各自对应2个文件夹内容，逐个进行操作
            for index in ["xx", "yy", "zz", "yz", "xz", "xy"]:
                # 定位计算的输出log目录，p结尾变量名对应施加+小应变的，n结尾变量名对应施加-小应变的。
                if config["calculator"] == "abacus":
                    output_log_p = os.path.join(efd, f"{index}+", "OUT.ABACUS", "running_scf.log")
                    output_log_n = os.path.join(efd, f"{index}-", "OUT.ABACUS", "running_scf.log")
                elif config["calculator"] == "lammps":
                    output_log_p = os.path.join(efd, f"{index}+", "log.lammps")
                    output_log_n = os.path.join(efd, f"{index}-", "log.lammps")
                else:
                    raise ValueError("Only support ABACUS or LAMMPS calculator")

                # 检查输出log文件是否存在。如果不存在，报error并退出程序
                if not os.path.exists(output_log_p) or not os.path.exists(output_log_n):
                    print("Error: Output log path for EFD tasks not found! Please run EFD tasks first!")
                    sys.exit(1)

                # 读取log文件中的总能量数据
                if config["calculator"] == "abacus":
                    Eij_p = parse_abacus_log(output_log_p)["final_etot"]
                    Eij_n = parse_abacus_log(output_log_n)["final_etot"]
                elif config["calculator"] == "lammps":
                    Eij_p = parse_lammps_log(output_log_p)["final_etot"]
                    Eij_n = parse_lammps_log(output_log_n)["final_etot"]
                else:
                    raise ValueError("Only support ABACUS or LAMMPS calculator")

                # 读取strain.json获得应变张量，并提取非零的数值存储为分量值。根据公式规范，这里应当取正值
                strain_tensor = loadfn(os.path.join(efd, f"{index}+", "strain.json"))
                # 用strain_tensor != 0创建布尔掩码，对应非零元素的位置
                nonzero_element = strain_tensor[strain_tensor != 0]
                # 应变张量是对称矩阵，取第1个非零值即可。用item()函数提取值返回float类型
                strain_value = nonzero_element[0].item()

                # 测试：添加对角项修正
                # if index == "xx" or index == "yy" or index == "zz":
                #     strain_value /= 1 + strain_value

                # 中心差分的应力计算公式：stress_value = (Eij_n - Eij_p) / (2*strain_value * volume)
                stress_value = (Eij_n - Eij_p) / (2 * strain_value) / volume
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
            if config["calculator"] == "abacus":
                stress_tensor *= ABACUS_UnitTrans
                # 对于计算得到的分量数值结果，数值过小的认为等于0。在KBar压强单位下最大容差设置为0.0001KBar
                cleaned_tensor = clean_matrix(stress_tensor, float_tolerance=1.0e-4)
            elif config["calculator"] == "lammps":
                stress_tensor *= LAMMPS_UnitTrans
                # 对于计算得到的分量数值结果，数值过小的认为等于0。在Bar压强单位下最大容差设置为0.1KBar
                cleaned_tensor = clean_matrix(stress_tensor, float_tolerance=0.1)
            else:
                raise ValueError("Only support ABACUS or LAMMPS calculator")

            # 将应力张量计算结果输出
            dumpfn(cleaned_tensor, "stress.json", indent=4)

            # 回到工作目录
            os.chdir(work_path)

    return


if __name__ == "__main__":
    __EFD_post__()
