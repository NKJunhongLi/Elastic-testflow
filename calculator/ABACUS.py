from Conf import Conf
import numpy as np


class ABACUS:
    def __init__(self, stru: Conf, parameters: dict):
        self.stru = stru
        self.parameters = parameters

    def make_input(self, cal_type, output_file):
        """
        生成INPUT文件并写入指定路径
        :param cal_type: 只允许输入"relax"或"scf"，分别用于生成relax或scf类型计算的INPUT文件
        :param output_file: 输出文件的路径
        :return: 纯操作，无返回值
        """
        # 构造一个输出模板
        template = ""

        template += "INPUT_PARAMETERS\n"

        template += "# Parameters (General)\n"
        template += "suffix    ABACUS\n"
        template += f"pseudo_dir    {self.parameters['pseudo_dir']}\n" if 'pseudo_dir' in self.parameters else ""
        template += f"orbital_dir    {self.parameters['orbital_dir']}\n" if 'orbital_dir' in self.parameters else ""
        template += f"nspin    {str(self.get_val('nspin', 1))}\n"
        template += f"symmetry    {str(self.get_val('symmetry', 1))}\n"
        template += f"esolver_type    ksdft\n"
        template += f"ks_solver    {'cg' if self.parameters['basis_type'] == 'pw' else 'genelpa'}\n"
        template += "\n"

        template += "# Parameters (Methods)\n"
        template += f"calculation    {cal_type}\n"
        if cal_type != "relax" or cal_type != "scf":
            raise ValueError("The function ABACUS.make_input() only support cal_type == 'relax' or 'scf'!")
        template += f"basis_type    {self.get_val('basis_type', 'lcao')}\n"
        template += f"ecutwfc    {str(self.get_val('ecutwfc', 100))}\n"
        template += f"scf_thr    {str(self.get_val('scf_thr', 1e-7))}\n"
        template += f"scf_nmax    {str(self.get_val('scf_nmax', 100))}\n"
        if cal_type == "relax":
            template += f"relax_method    {self.get_val('relax_method', 'cg')}\n"
            template += f"relax_nmax    {str(self.get_val('relax_nmax', 100))}\n"
            template += f"force_thr_ev    {str(self.get_val('force_thr_ev', 0.01))}\n"
            template += f"stress_thr    {str(self.get_val('stress_thr', 0.5))}\n"
            template += f"chg_extrap    {self.get_val('chg_extrap', 'first-order')}\n"

        # 设置泛函部分
        template += f"dft_functional    {self.parameters['dft_functional']}\n" \
            if 'dft_functional' in self.parameters else ""

        template += "\n"

        # Elastic性质计算主要关心应力计算
        template += "# Parameters (File)\n"
        template += "cal_force    1\n"
        template += "cal_stress    1\n"
        template += "\n"

        template += "# Parameters (smearing)\n"
        template += f"smearing_method    {self.get_val('smearing_method', 'gaussian')}\n"
        template += f"smearing_sigma    {str(self.get_val('smearing_sigma', 0.01))}\n"
        template += "\n"

        template += "# Parameters (K points spacing)\n"
        template += f"kspacing    {str(self.get_val('kspacing', 0.1))}\n"
        template += "\n"

        # 设置mixing_beta部分
        template += "# Parameters (mixing)\n"
        template += f"mixing_type    {self.get_val('mixing_type', 'broyden')}\n"
        if 'mixing_beta' in self.parameters:
            template += f"mixing_beta    {str(self.parameters['mixing_beta'])}\n"
        elif self.parameters["nspin"] == 2 or self.parameters["nspin"] == 4:
            template += "mixing_beta    0.4\n"
        else:
            template += "mixing_beta    0.8\n"

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


def parse_abacus_log(logfile):
    """解析abacus的输出log，提取应力张量和总能量，保存在一个字典里。单位分别为为KBAR和eV"""
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

    # 遍历文件内容，寻找信息
    for ii in range(len(lines)):
        # 如果找到总能量标题，解析总能量信息
        if not found_energy and "!FINAL_ETOT_IS" in lines[ii]:
            try:
                parts = lines[ii].strip().split()
                # log里总能量输出格式为： !FINAL_ETOT_IS $(etot_number) eV，总能量的文本对应为第2个。
                etot_str = parts[1]
                output_dict["final_etot"] = float(etot_str)
                found_energy = True
                continue
            except (IndexError, ValueError) as e:
                print(f"Error while parsing energy value: {e}")
                continue

        # 寻找stress标题
        elif "TOTAL-STRESS (KBAR)" not in lines[ii]:
            continue

        # 找到stress标题，开始收集数据
        else:
            try:
                jj = ii + 1
                matrix = []

                # 扫描后续5行
                while jj <= min(ii + 5, len(lines)):
                    line = lines[jj].strip()
                    # 跳过分割线
                    if line.startswith("--"):
                        jj += 1
                        continue
                    parts = line.split()
                    data = [float(kk) for kk in parts]
                    matrix.append(data)
                    jj += 1

                # 更新stress结果。代码循环执行到最后，存储的是最后一步stress输出的结果
                output_dict["stress"] = np.array(matrix)
            except (IndexError, ValueError) as e:
                print(f"Error while parsing stress: {e}")
                continue

    if not found_energy:
        print("Warning! Did not find final energy.")

    return output_dict
