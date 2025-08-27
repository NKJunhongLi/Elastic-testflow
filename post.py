import os
import sys
import numpy as np

from monty.serialization import loadfn, dumpfn
from glob import glob
from calculator.ABACUS import parse_abacus_log
from calculator.LAMMPS import parse_lammps_log
from make import clean_matrix


def set_voigt(matrix: np.array((3, 3))):
    """
    把一个对称的3×3矩阵转换为voigt notation下的6×1列向量
    :param matrix: 一个对称的3×3的numpy矩阵
    :return: 一个6×1的numpy矩阵
    """
    voigt_result = np.zeros((6, 1))

    voigt_result[0] = matrix[(0, 0)]
    voigt_result[1] = matrix[(1, 1)]
    voigt_result[2] = matrix[(2, 2)]
    voigt_result[3] = matrix[(1, 2)]
    voigt_result[4] = matrix[(0, 2)]
    voigt_result[5] = matrix[(0, 1)]

    return voigt_result


def print_result(C: np.array((6, 6)), output_filename):
    """
    根据拟合得到的弹性模量矩阵计算体积模量、杨氏模量、剪切模量、泊松比，并打印数据
    :param output_filename: 输出的.txt目标文件路径
    :param C: 6×6的numpy矩阵，存储弹性模量矩阵
    """
    # 计算体积模量
    Bv = 0.0
    for ii in range(3):
        for jj in range(3):
            Bv += C[(ii, jj)] / 9

    # 计算剪切模量
    Gv = (
            ((C[(0, 0)] + C[(1, 1)] + C[(2, 2)])
             - (C[(0, 1)] + C[(0, 2)] + C[(1, 2)])
             + 3 * (C[(3, 3)] + C[(4, 4)] + C[(5, 5)]))
            / 15
    )

    # 计算杨氏模量
    Ev = 9 * Bv * Gv / (3 * Bv + Gv)

    # 计算泊松比
    Muv = (3 * Bv - 2 * Gv) / (3 * Bv + Gv) / 2

    # 构造输出模板，填入内容
    template = ""
    template += "Elastic constant (GPa): \n"
    template += "=" * 30 + "\n\n"

    for row in range(6):
        for column in range(6):
            template += f"{C[(row, column)]:10.2f}"  # 占10位字符，保留2位小数
        template += "\n"

    template += "\n" + "=" * 30 + "\n\n"

    template += f"Bulk Modulus Bv = {Bv:.2f} (GPa)\n"
    template += f"Youngs Modulus Ev = {Ev:.2f} (GPa)\n"
    template += f"Shear Modulus Gv = {Gv:.2f} (GPa)\n"
    template += f"Poisson Ratio = {Muv:.2f}\n"
    # 写入文本文件
    with open(output_filename, 'w') as f:
        f.write(template)

    return


def post():
    # 从命令行中获得配置文件，如果没有给，报error并退出程序。
    if len(sys.argv) < 2:
        print("Usage: python3 post.py $(your configuration filename)")
        print("Error: Please provide the configuration file")
        sys.exit(1)

    # 加载配置文件，使用monty.serialization.loadfn()以获得更高兼容性
    config = loadfn(sys.argv[1])

    if config["calculator"] not in ["abacus", "lammps"]:
        raise ValueError("Only support ABACUS or LAMMPS calculator")

    cwd = os.getcwd()
    # make步骤的工作文件夹为work
    work_path = os.path.join(cwd, "work")
    if not os.path.exists(work_path):
        print("Error: Work path not found! Please do make first!")
        sys.exit(1)

    # make前提供的初始结构文件路径，存在一个list里
    origin_structures = [os.path.join(cwd, file) for file in config["stru_files"]]

    # 这个变量用于访问config字典中"relax_log_dir"对应的list元素
    strufile_index: int = 0

    for origin_strufile in origin_structures:
        # 记录结构文件的文件名，用于接下来进入"work"目录下的结构操作文件夹
        stru_filename = os.path.basename(origin_strufile)

        os.chdir(os.path.join(work_path, stru_filename))

        # 定义一个bool型变量，用于后续代码中是否进行了能量差分的判断
        EFD_flag: bool = False

        # 判断是否有进行能量差分操作，如果有，设置EFD_flag = True，且后续读取能量差分得到的stress数据
        if os.path.exists(os.path.join(work_path, stru_filename, "EFD")):
            EFD_flag = True

        # 声明两个list型变量用来存储24组数据，stress存储应力，strains存储应变，以voigt notation的6×1的numpy矩阵形式存储
        stresses_list = []
        strains_list = []

        # 定位初始结构结构优化的输出log，并解析log文件得到初始应力；变胞结构应力要减去初始应力得到净应力再用于拟合
        origin_output_log = os.path.join(cwd, config["relax_log_dir"][strufile_index])
        if config["calculator"] == "abacus":
            origin_stress = parse_abacus_log(origin_output_log)["stress"]
        elif config["calculator"] == "lammps":
            origin_stress = parse_lammps_log(origin_output_log)["stress"]
        else:
            raise ValueError("Only support ABACUS or LAMMPS calculator")

        # 如果进行了能量差分，初始应力使用EFD_task里差分得到的数据
        if EFD_flag:
            origin_stress = loadfn(os.path.join(work_path, stru_filename, "EFD_task", "stress.json"))

        # 使用glob()函数找到24个task文件夹并存到一个list里
        task_dirs = sorted(glob(os.path.join(work_path, stru_filename, "task.*")))

        # 遍历所有task文件夹，收集应力和应变数据
        for task in task_dirs:
            # 读取应变strain数据，转换为voigt notation，并存入list变量strains里
            strain_matrix = loadfn(os.path.join(task, "strain.json"))
            strain_voigt = set_voigt(strain_matrix)
            strains_list.append(strain_voigt)

            # 定位relax的输出log，并解析log文件获得应力stress矩阵
            if config["calculator"] == "abacus":
                task_output_log = os.path.join(task, "OUT.ABACUS", "running_relax.log")
                stress_matrix = parse_abacus_log(task_output_log)["stress"]
            elif config["calculator"] == "lammps":
                task_output_log = os.path.join(task, "log.lammps")
                stress_matrix = parse_lammps_log(task_output_log)["stress"]
            else:
                raise ValueError("Only support ABACUS or LAMMPS calculator")

            # 如果进行了能量差分，应力使用EFD里差分得到的数据
            if EFD_flag:
                stress_matrix = loadfn(os.path.join(task, "EFD", "stress.json"))

            # 应力减去初始应力得到净应力
            net_stress = stress_matrix - origin_stress
            # 转换为voigt notation，并存入list变量stresses里
            stress_voigt = set_voigt(net_stress)
            stresses_list.append(stress_voigt)

        # 把list里存的数据合并成一个大的6×24的numpy矩阵
        strains_matrix = np.hstack(strains_list)
        stresses_matrix = np.hstack(stresses_list)

        # stress的值全部乘以-1，以使得输出的正负号与物理约定一致
        stresses_matrix *= -1.0

        # 定义6×6的numpy矩阵用来存储弹性模量矩阵
        C = np.zeros((6, 6))

        # 遍历6个应力分量方向，逐行进行弹性模量矩阵的拟合
        for i in range(6):
            A = strains_matrix.transpose()  # 为匹配numpy.linalg.lstsq()函数规范，构造设计矩阵A为24×6形状
            b = stresses_matrix.transpose()[:, i]  # 为匹配numpy.linalg.lstsq()函数规范，构造观测矩阵A为24×1形状
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # 使用numpy.linalg.lstsq()函数保证拟合截距为0
            C[i, :] = x  # 一次性拟合得到了6个分量的结果，赋值到一行里面

        # 单位转换
        if config["calculator"] == "abacus":
            C /= 10.0  # 对于ABACUS，从KBar到GPa
        elif config["calculator"] == "lammps":
            C /= 10000.0  # 对于LAMMPS，从Bar到GPa
        else:
            raise ValueError("Only support ABACUS or LAMMPS calculator")

        # 对小于0.01GPa的结果，认为是数值误差，清除
        cleaned_C = clean_matrix(C, float_tolerance=0.01)

        # 新建一个elastic_constant文件夹用于保存输出结果
        os.makedirs("elastic_constant", exist_ok=True)

        # 将弹性模量矩阵保存为.json文件
        dumpfn(cleaned_C, os.path.join(os.getcwd(), "elastic_constant", "elastic_constant.json"), indent=4)

        # 保存弹性模量矩阵、体积模量、杨氏模量、剪切模量、泊松比到.txt文件
        print_result(cleaned_C, os.path.join(os.getcwd(), "elastic_constant", "result.txt"))

        strufile_index += 1

    return


if __name__ == "__main__":
    post()
