import os
import numpy as np


class LAMMPS:
    def __init__(self, conf_file, parameters: dict, interaction: dict):
        self.conf_file = os.path.basename(conf_file)
        self.parameters = parameters
        self.interaction = interaction

    def make_in(self, cal_type, output_file):
        """
        生成LAMMPS的输入文件并写入指定路径
        :param cal_type: 只允许输入"minimize"或"run_zero"，分别用于生成结构优化或单点计算计算的in.lammps文件
        :param output_file: 输出文件的路径
        :return: 纯操作，无返回值
        """
        # 构造一个输出模板
        template = ""

        # 模拟基础设置
        template += "clear\n"
        template += "units    metal\n"  # 设置金属单位制，与Conf.py中使用的单位匹配
        template += "dimension    3\n"
        template += "boundary    p p p\n"  # 循环边界
        template += "box    tilt large\n"
        template += "\n"

        # 读取结构文件
        template += f"read_data    {self.conf_file}\n"
        template += "\n"

        template += "neigh_modify    every 1 delay 0 check no\n"
        template += "\n"

        # 相互作用，读取深度势能模型的情况
        if self.interaction["method"] == "deepmd":
            template += f"pair_style    deepmd    {self.interaction['model']}\n"
            template += "pair_coeff * *\n"
            template += "\n"

        # 物理量的计算设置，以及输出设置
        template += "compute    mype all pe\n"
        template += "thermo    100\n"
        template += "thermo_style    custom step pe pxx pyy pzz pxy pxz pyz lx ly lz vol c_mype\n"
        template += "dump    1 all custom 100 dump.relax id type xs ys zs fx fy fz\n"
        template += "write_data    CONTCAR.lmp\n"
        template += "\n"

        # relax参数设置
        if cal_type == "minimize":
            template += "min_style    cg\n"
            template += (
                f"minimize    "
                f"{self.get_val('etol', 0.0)} "
                f"{self.get_val('ftol', 1e-10)} "
                f"{self.get_val('maxiter', 5000)} "
                f"{self.get_val('maxeval', 500000)}\n"
            )

        elif cal_type == "run_zero":
            template += "run    0\n"

        else:
            raise ValueError("The function LAMMPS.make_in() only support cal_type == 'minimize' or 'run_zero'!")

        template += "\n"

        # 计算能量和压强（应力）
        template += "variable    E equal \"c_mype\"\n"
        template += "variable    Pxx equal pxx\n"
        template += "variable    Pyy equal pyy\n"
        template += "variable    Pzz equal pzz\n"
        template += "variable    Pyz equal pyz\n"
        template += "variable    Pxz equal pxz\n"
        template += "variable    Pxy equal pxy\n"
        template += "\n"

        # 屏幕和log中的输出
        template += "print \"All done\"\n"
        template += "print \"Final energy = ${E}\"\n"
        template += "print \"Final Stress (xx yy zz yz xz xy) = ${Pxx} ${Pyy} ${Pzz} ${Pyz} ${Pxz} ${Pxy}\""
        template += "\n"

        # 将模板写入到文件
        with open(output_file, 'w') as f:
            f.write(template)

        return

    def get_val(self, param, default_val):
        """判断配置文件的"parameters"部分是否有对应的参数设置。如果有，返回设置值；如果没有，返回默认值"""
        if param in self.parameters:
            return self.parameters[param]
        else:
            return default_val


def parse_lammps_log(logfile):
    """解析lammps的输出log，提取应力张量和总能量，保存在一个字典里。单位分别为为BAR和eV"""
    # 初始化一个字典用于结果输出
    output_dict = {"stress": np.zeros((3, 3)), "final_etot": 0.0}

    # 尝试打开并读取log文件，并把内容保存在一个list里，list里一个元素即是文件里的一行
    try:
        with open(logfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error! File {logfile} not exist!")
        return output_dict
    except Exception as e:
        print(f"Error while reading logfile: {e}")
        return output_dict

    # 标志变量，用于标记是否找到关键行
    found_energy = False
    found_stress = False

    # 逐行解析文件内容
    for line in lines:
        # 去除行首尾的空白字符
        stripped_line = line.strip()

        # 检查是否找到最终能量行
        if not found_energy and stripped_line.startswith("Final energy ="):
            try:
                # 分割字符串获取能量值
                parts = stripped_line.split()
                energy = float(parts[-1])
                output_dict["final_etot"] = energy
                found_energy = True
            except (IndexError, ValueError) as e:
                print(f"Error while parsing energy value: {e}")
                continue

        # 检查是否找到应力行
        elif not found_stress and stripped_line.startswith("Final Stress (xx yy zz yz xz xy) ="):
            try:
                # 分割字符串获取应力分量
                parts = stripped_line.split()
                # 应力分量的顺序为xx yy zz yz xz xy
                stress_xx = float(parts[9])  # 第一个应力分量
                stress_yy = float(parts[10])  # 第二个应力分量
                stress_zz = float(parts[11])  # 第三个应力分量
                stress_yz = float(parts[12])  # 第四个应力分量 (yz)
                stress_xz = float(parts[13])  # 第五个应力分量 (xz)
                stress_xy = float(parts[14])  # 第六个应力分量 (xy)

                # 填充3x3应力张量
                stress = np.zeros((3, 3))
                stress[0, 0] = stress_xx  # xx分量
                stress[1, 1] = stress_yy  # yy分量
                stress[2, 2] = stress_zz  # zz分量
                stress[1, 2] = stress_yz  # yz分量
                stress[2, 1] = stress_yz  # zy分量 (对称)
                stress[0, 2] = stress_xz  # xz分量
                stress[2, 0] = stress_xz  # zx分量 (对称)
                stress[0, 1] = stress_xy  # xy分量
                stress[1, 0] = stress_xy  # yx分量 (对称)

                output_dict["stress"] = stress
                found_stress = True
            except (IndexError, ValueError) as e:
                print(f"Error while parsing stress: {e}")
                continue

        # 如果两个值都已找到，可以提前退出循环
        if found_energy and found_stress:
            break

    # 检查是否成功找到所有数据
    if not found_energy:
        print("Warning! Did not find final energy.")
    if not found_stress:
        print("Warning! Did not find stress")

    return output_dict
