import os
import sys

from copy import deepcopy
from glob import glob
from shutil import copy
from monty.serialization import loadfn, dumpfn
from make import create_workpath, clean_matrix
from Conf import Conf
from StrainTensor import KittelStrain, make_small_deforms
from calculator.ABACUS import ABACUS
from calculator.LAMMPS import LAMMPS


def __EFD_make__():
    """
    在执行完make并运行完成relax任务后，对24个task变胞结构和初始结构生成微小应变结构，用于scf计算做能量差分。\n
    每个task文件夹内新建EFD文件夹，EFD文件夹内新建12个文件夹存储对应分量的微小应变结构，命名为 "xx+" "xx-" ... "zz+" "zz-"\n
    12文件夹每个都可直接运行abacus。\n
    在操作目录下新建EFD_task文件夹，与EFD文件夹结构一样，存储初始结构的微小应变结构。
    """
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 EFD_make.py $(your configuration filename)")
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

    # 存储6个变形张量的字典。如果配置文件中设置了步长，使用设置值，否则使用默认值。
    if "small_deform" in config:
        small_deforms = make_small_deforms(config["small_deform"])

    else:
        small_deforms = make_small_deforms()

    # make前提供的初始结构文件路径，存在一个list里
    origin_structures = [os.path.join(cwd, file) for file in config["stru_files"]]

    # 对make步骤中操作过的每个结构的文件夹一一进行操作
    for origin_strufile in origin_structures:
        # 记录结构文件的文件名，用于接下来进入"work"目录下的结构操作文件夹
        stru_filename = os.path.basename(origin_strufile)

        # 使用glob()函数找到24个task文件夹并存到一个list里
        task_dirs = sorted(glob(os.path.join(work_path, stru_filename, "task.*")))
        # 把操作目录也添加到list里，在操作目录下准备初始结构的微小应变任务
        task_dirs.append(os.path.join(work_path, stru_filename))

        # 对task文件夹里relax完得到的结构，逐个进行微小应变任务的准备
        for task in task_dirs:
            # 进入task文件夹
            os.chdir(task)

            # 检查make之后有没有对task文件夹里结构进行relax，如果没有，报error并退出程序
            if os.path.basename(task) == "task.*":
                if config["calculator"] == "abacus" and not os.path.exists(os.path.join(task, "OUT.ABACUS")):
                    print("Error: ABACUS output path not found! Please run relax tasks first!")
                    sys.exit(1)

                if config["calculator"] == "lammps" and not os.path.exists(os.path.join(task, "log.lammps")):
                    print("Error: LAMMPS output log path not found! Please run relax tasks first!")
                    sys.exit(1)

            # 在task文件夹里新建一个EFD文件夹用于能量差分计算。自动备份旧的EFD文件夹
            EFD_path = create_workpath(os.path.join(task, "EFD"))
            # 对于初始结构的操作，新建文件夹命名为"EFD_task"
            if os.path.basename(task) == stru_filename:
                EFD_path = create_workpath(os.path.join(task, "EFD_task"))

            # 进入EFD文件夹
            os.chdir(EFD_path)

            INPUT = os.path.join(os.getcwd(), "INPUT")  # str型变量，记录ABACUS输入文件的路径
            in_lammps = os.path.join(os.getcwd(), "in.lammps")  # str型变量，记录LAMMPS输入文件的路径
            # 如果有使用deempd势函数模型，定义一个str型变量记录模型文件路径
            model = os.path.join(cwd, config["interaction"]["model"]) \
                if config["interaction"]["method"] == "deepmd" else None

            if config["calculator"] == "abacus":
                # 定位relax完的结构文件STRU_ION_D
                relaxed_stru = os.path.join(task, "OUT.ABACUS", "STRU_ION_D")
                # 对于初始结构的操作，结构文件是配置文件给出的
                if os.path.basename(task) == stru_filename:
                    relaxed_stru = origin_strufile
                # 读取结构文件
                relaxed_conf = Conf.from_abacus(relaxed_stru)
                # 把STRU文件写入EFD文件夹中
                relaxed_conf.to_abacus(os.path.join(os.getcwd(), "STRU"))

                # 用relax完的结构和配置文件里的"parameters"初始化ABACUS类作为计算器
                cal = ABACUS(config["parameters"])
                # 对6个分量各自的scf计算，INPUT文件都一样。将它先写入EFD文件夹里，后续再复制到对分量计算操作的文件夹里
                cal.make_input("scf", INPUT)

            elif config["calculator"] == "lammps":
                # 定位relax完的结构文件CONTCAR.lmp
                relaxed_stru = os.path.join(task, "CONTCAR.lmp")
                # 对于初始结构的操作，结构文件是配置文件给出的
                if os.path.basename(task) == stru_filename:
                    relaxed_stru = origin_strufile
                # 读取结构文件
                relaxed_conf = Conf.from_lammps(relaxed_stru, config["interaction"]["type_map"])
                # 把conf.lmp文件写入EFD文件夹中
                relaxed_conf.to_lammps(os.path.join(os.getcwd(), "conf.lmp"), config["interaction"]["type_map"])

                # 用relax完的结构文件，以及配置文件里的"parameters"与"interaction"两项设置，初始化LAMMPS类作为计算器
                cal = LAMMPS(os.path.join(os.getcwd(), "conf.lmp"), config["parameters"], config["interaction"])
                # 对6个分量各自的单点run 0计算，in文件设置都一样。将它先写入EFD文件夹里，后续再复制到对分量计算操作的文件夹里
                cal.make_in("run_zero", in_lammps)

            else:
                raise ValueError("Only support ABACUS or LAMMPS calculator")

            # 对6个应变分量方向分别进行操作，一共生成12组文件
            for index in small_deforms:
                # 每个分量单独新建一个文件夹，并进入该文件夹
                os.makedirs(index)
                os.chdir(os.path.join(EFD_path, index))

                # 将对应的变形张量作用于结构，并将变形后结构写入文件夹内
                tmp_conf = deepcopy(relaxed_conf)
                tmp_conf.apply_deform(small_deforms[index])
                if config["calculator"] == "abacus":
                    tmp_conf.to_abacus(os.path.join(os.getcwd(), "STRU"))
                    copy(INPUT, os.getcwd())
                elif config["calculator"] == "lammps":
                    # 为了与in.lammps里设置的read_data统一，文件名必须是conf.lmp
                    tmp_conf.to_lammps(os.path.join(os.getcwd(), "conf.lmp"), config["interaction"]["type_map"])
                    copy(in_lammps, os.getcwd())
                    if config["interaction"]["method"] == "deepmd":
                        copy(model, os.getcwd())
                else:
                    raise ValueError("Only support ABACUS or LAMMPS calculator")

                # 从变形张量计算获取对应的应变张量
                current_strain = KittelStrain.from_deform(small_deforms[index])
                # 新建一个numpy矩阵用于存储清除浮点误差后的应变张量
                cleaned_matrix = clean_matrix(current_strain.matrix)
                # 将结果写入.json文件中。使用monty.serialization.dumpfn()以获得更高兼容性
                dumpfn(cleaned_matrix, "strain.json", indent=4)
                # 顺便将变形矩阵也写入一个.json文件
                dumpfn(small_deforms[index].deform_matrix, "deform.json", indent=4)

                # 回到EFD文件夹
                os.chdir(EFD_path)

    return


if __name__ == "__main__":
    __EFD_make__()
