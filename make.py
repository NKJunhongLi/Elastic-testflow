import os
import sys
import re
import numpy as np

from copy import deepcopy
from shutil import copy
from monty.serialization import loadfn, dumpfn
from calculator.ABACUS import ABACUS
from calculator.LAMMPS import LAMMPS
from Conf import Conf
from StrainTensor import KittelStrain, make_norm_deforms, make_shear_deforms


def clean_matrix(matrix: np.array, float_tolerance=1.0e-12):
    """
    将给定的numpy数值矩阵按照需要的精度去除浮点误差
    :param matrix: 输入要进行清除浮点误差操作的numpy矩阵
    :param float_tolerance: 设置最大容许误差，默认值为1.0e-12
    :return: 清除浮点误差后的numpy矩阵
    """
    result = np.array(
        # 嵌套list推导式，清除浮点误差
        [
            [
                # 对numpy内存储的每个值进行判断，若绝对值小于tolerance，则赋值为0
                0.0 if abs(num) < float_tolerance else num
                for num in row
            ]
            for row in matrix
        ]
    )
    return result


def create_workpath(pathname: str) -> str:
    """创建新工作目录。如果文件夹重名，自动备份旧文件夹，并新建空的新文件夹"""
    # 处理旧工作目录
    if os.path.exists(pathname) and os.path.isdir(pathname):
        max_num = -1  # 初始化备份编号
        pattern = re.compile(r'^bk(\d+)_' + re.escape(pathname) + r'$')  # 匹配组(0)为'bk**_'，匹配组(1)为'bk'后的数字

        # 遍历当前目录寻找已有备份
        for name in os.listdir('.'):
            # 只处理文件夹
            if os.path.isdir(name):
                match = pattern.match(name)
                if match:
                    # 对应匹配组(1)，提取捕获到的文件夹中的数字，并强制转换类型为int
                    current_num = int(match.group(1))
                    max_num = max(max_num, current_num)

        # 生成新备份号（自动递增）
        new_num = max_num + 1
        while True:
            # 格式化备份的目录名
            new_backup = f"bk{new_num:02d}_{os.path.basename(pathname)}"  # 保持至少两位数
            if not os.path.exists(new_backup):
                break
            new_num += 1

        # 执行重命名
        os.rename(pathname, new_backup)
        print(f"Have already renamed the old workpath as {new_backup}")

    # 创建新工作目录
    os.makedirs(pathname, exist_ok=True)
    print("Create new folder ", pathname, " as workpath")
    return os.path.join(os.getcwd(), pathname)


def __make__():
    """
    根据配置文件内的设置，对提供的结构文件施加应变，输出变胞后的STRU，并根据配置文件内的参数生成INPUT文件。\n
    在work目录下对每个提供的结构文件创建操作目录，并在操作目录内建好24个task文件夹。\n
    每个task文件夹内存有STRU文件，INPUT文件和对应的应变张量strain.json文件。每个task文件夹下都可直接运行abacus。
    """
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 make.py $(your configuration filename)")
        print("Error: Please provide the configuration file")
        sys.exit(1)

    # 加载配置文件，使用monty.serialization.loadfn()以获得更高兼容性
    config = loadfn(sys.argv[1])

    if config["calculator"] not in ["abacus", "lammps"]:
        raise ValueError("Only support ABACUS or LAMMPS calculator")

    # 创建工作目录
    cwd = os.getcwd()
    work_path = create_workpath(os.path.join(cwd, "work"))

    # 进入工作目录
    os.chdir(work_path)

    norm_deform_list = make_norm_deforms(config["norm_deform"])
    shear_deform_list = make_shear_deforms(config["shear_deform"])

    # 结构文件路径。将多个文件存在一个list里
    structures = [os.path.join(cwd, file) for file in config["stru_files"]]

    for file in structures:
        # 对每个结构文件单独建一个文件夹用于操作
        stru_name = os.path.basename(file)
        os.makedirs(stru_name)

        # 先进入操作结构的文件夹
        os.chdir(os.path.join(work_path, stru_name))

        INPUT = os.path.join(work_path, stru_name, "INPUT")  # str型变量，记录ABACUS输入文件的路径
        in_lammps = os.path.join(work_path, stru_name, "in.lammps")  # str型变量，记录LAMMPS输入文件的路径
        # 如果有使用deempd势函数模型，定义一个str型变量记录模型文件路径
        model = os.path.join(cwd, config["interaction"]["model"]) \
            if config["calculator"] == "lammps" and config["interaction"]["method"] == "deepmd" else None

        if config["calculator"] == "abacus":
            original_conf = Conf.from_abacus(file)
            # 把cell-relax完的结构先写入到操作文件夹里
            original_conf.to_abacus(os.path.join(work_path, stru_name, "STRU"))

            # 用平衡结构和配置文件里的"parameters"初始化ABACUS类作为计算器
            cal = ABACUS(config["parameters"])
            # 对每一次单点计算，INPUT文件都一样。INPUT先写入到操作文件夹里。后续再复制到任务文件夹里
            cal.make_input("relax", INPUT)

        elif config["calculator"] == "lammps":
            original_conf = Conf.from_lammps(file, config["interaction"]["type_map"])
            # 把cell-relax完的结构先写入到操作文件夹里，命名为conf.lmp
            original_conf.to_lammps(os.path.join(work_path, stru_name, "conf.lmp"), config["interaction"]["type_map"])

            # 用新写的平衡结构文件，以及配置文件里的"parameters"与"interaction"两项设置，初始化LAMMPS类作为计算器
            cal = LAMMPS(os.path.join(work_path, stru_name, "conf.lmp"), config["parameters"], config["interaction"])
            # 对每一次单点计算，输入文件in.lammps都一样。先写入到操作文件夹里。后续再复制到任务文件夹里
            cal.make_in("minimize", in_lammps)

            # 如果用dp模型计算，把模型文件也复制进来
            if config["interaction"]["method"] == "deepmd":
                copy(model, os.getcwd())

        else:
            raise ValueError("Only support ABACUS or LAMMPS calculator")

        # 分别进行24个变胞操作
        suffix: int = 1
        for tasks in [norm_deform_list, shear_deform_list]:
            for deform in tasks:
                # 为每一个变胞结构单独新建文件夹
                task_dir_name = f"task.{suffix:03d}"
                os.makedirs(task_dir_name, exist_ok=True)

                # 进入变胞任务的文件夹
                os.chdir(task_dir_name)

                if config["calculator"] == "abacus":
                    # 将变形作用于结构，并将变胞结构写入到文件夹内的STRU文件
                    tmp_conf = deepcopy(original_conf)
                    deformed_conf = tmp_conf.apply_deform(deform)
                    deformed_conf.to_abacus(os.path.join(work_path, stru_name, task_dir_name, "STRU"))

                    # 将上一级目录里的INPUT文件复制到这里
                    copy(INPUT, os.getcwd())

                elif config["calculator"] == "lammps":
                    # 将变形作用于结构，并将变胞结构写入到文件夹内，命名为conf.lmp文件
                    tmp_conf = deepcopy(original_conf)
                    deformed_conf = tmp_conf.apply_deform(deform)
                    deformed_conf.to_lammps(
                        os.path.join(work_path, stru_name, task_dir_name, "conf.lmp"),
                        config["interaction"]["type_map"]
                    )

                    # 将上一级目录里的in.lammps文件复制到这里
                    copy(in_lammps, os.getcwd())

                    # 如果用dp模型计算，把模型文件也复制进来
                    if config["interaction"]["method"] == "deepmd":
                        copy(model, os.getcwd())

                else:
                    raise ValueError("Only support ABACUS or LAMMPS calculator")

                # 从变形张量计算获取对应的应变张量
                current_strain = KittelStrain.from_deform(deform)
                # 新建一个numpy矩阵用于存储清除浮点误差后的应变张量
                cleaned_matrix = clean_matrix(current_strain.matrix)
                # 将结果写入.json文件中。使用monty.serialization.dumpfn()以获得更高兼容性
                dumpfn(cleaned_matrix, "strain.json", indent=4)

                # 返回上一级目录
                os.chdir(os.path.pardir)

                suffix += 1

        # 退出文件夹，回到work目录
        os.chdir(work_path)

    return


if __name__ == "__main__":
    __make__()
