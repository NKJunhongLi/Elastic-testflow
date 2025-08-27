import numpy as np
from PeriodicTable import get_mass
from StrainTensor import DeformationTensor

# 单位转换系数：1.0 bohr = 0.529177210544 Angstrom
Bohr2Angstrom = 0.529177210544


class Conf:
    """
    存储结构信息的类，包括晶格矢量、原子种类、原子坐标。\n
    对于ABACUS计算，还会存储赝势文件、轨道文件、自旋极化强度。\n
    内置函数功能有：\n
    从ABACUS的STRU结构文件中读取信息构造类对象\n
    从LAMMPS的.lmp格式结构文件中读取信息构造类对象\n
    将结构信息输出到STRU格式文件\n
    将结构信息输出到.lmp格式文件\n
    对结构施加应变
    """
    def __init__(self):
        """Conf类的构造函数。确定成员变量"""
        # 晶格常数（单位转换因子）。本程序统一以Angstrom作为距离单位。
        self.lattice_constant = 1.0

        # 晶格矢量矩阵（3x3 numpy数组，存储实际物理量）。数值以Angstrom单位存储。
        self.lattice_vectors = np.zeros((3, 3))

        # 坐标类型：Direct（分数坐标）或 Cartesian（笛卡尔坐标）
        self.coordinate_type = "Cartesian"

        # 原子种类顺序列表（维护输入文件中的元素顺序）
        self.atomic_species = []

        # 原子质量字典（键：元素符号，值：质量字符串）
        self.masses = {}

        # 原子自旋磁矩字典（键：元素符号，值：磁矩）
        self.magnetism = {}

        # 赝势文件字典（键：元素符号，值：赝势文件名）
        self.pseudo_files = {}

        # 轨道文件字典（键：元素符号，值：轨道文件名）
        self.orbital_files = {}

        # 原子位置字典（键：元素符号，值：存储坐标的numpy矩阵）
        self.atomic_positions = {}

    @classmethod
    def from_abacus(cls, stru_file):
        """从ABACUS结构文件构造对象"""
        instance = cls()

        with open(stru_file, 'r') as f:
            current_section = None
            for line in f:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue

                # 检测章节标签
                if line in ["ATOMIC_SPECIES", "NUMERICAL_ORBITAL",
                            "LATTICE_CONSTANT", "LATTICE_VECTORS",
                            "ATOMIC_POSITIONS"]:
                    current_section = line
                    continue

                # 解析原子种类部分
                if current_section == "ATOMIC_SPECIES":
                    parts = line.split()
                    element = parts[0]
                    instance.atomic_species.append(element)
                    instance.masses[element] = parts[1]
                    instance.pseudo_files[element] = parts[2]

                # 解析轨道文件部分
                elif current_section == "NUMERICAL_ORBITAL":
                    # 轨道文件顺序与atomic_species一致
                    element = instance.atomic_species[len(instance.orbital_files)]
                    instance.orbital_files[element] = line

                # 解析晶格常数
                elif current_section == "LATTICE_CONSTANT":
                    instance.lattice_constant = float(line)

                # 解析晶格矢量
                elif current_section == "LATTICE_VECTORS":
                    # 读取连续三行构成3x3矩阵
                    for lv in range(3):
                        instance.lattice_vectors[lv] = np.array(list(map(float, line.split())))
                        line = next(f).strip()
                    # 实际晶格矢量 = 文件数值 × 转换系数，单位为bohr
                    # 注意abacus的长度单位是bohr，所以转换系数还要乘以系数0.529177210544，才是Angstrom单位
                    instance.lattice_vectors *= instance.lattice_constant * Bohr2Angstrom

                # 解析原子位置
                elif current_section == "ATOMIC_POSITIONS":
                    # 第一行为坐标类型
                    if line == "Direct" or line == "Cartesian":
                        instance.coordinate_type = line
                        continue

                    # 下面处理原子信息部分

                    # 第1行是元素类型
                    element = line.split()[0]

                    # 第2行是极化强度
                    magnetism_line = next(f).strip()
                    magnetism = float(magnetism_line.split()[0])
                    instance.magnetism[element] = magnetism

                    # 第3行是对应元素的原子数N
                    atom_num_line = next(f).strip()
                    atom_num = int(atom_num_line.split()[0])

                    # 第4行开始是原子坐标
                    positions = np.zeros((atom_num, 3))
                    for nn in range(atom_num):
                        position_line = next(f).strip()
                        coord = [position_line.split()[i] for i in range(3)]
                        positions[nn] = np.array(coord)
                    instance.atomic_positions[element] = positions

        return instance

    def to_abacus(self, output_file):
        """输出ABACUS格式结构文件"""
        # 构造一个输出模板
        template = ""

        # 原子种类部分
        template += "ATOMIC_SPECIES\n"
        for element in self.atomic_species:
            template += f"{element} {self.masses[element]} {self.pseudo_files[element]}\n"
        template += "\n"

        # 原子轨道文件部分。对于pw计算，STRU文件中可以没有轨道文件
        if len(self.orbital_files) != 0:
            template += "NUMERICAL_ORBITAL\n"
            for element in self.atomic_species:
                template += f"{self.orbital_files[element]}\n"
            template += "\n"

        # 晶格常数系数部分
        template += "LATTICE_CONSTANT\n"
        template += f"{self.lattice_constant:.10f}\n"
        template += "\n"

        # 晶格矢量部分
        template += "LATTICE_VECTORS\n"
        # 3×3矩阵，逐个元素打印，打印3个后换行。控制输出精度为小数点后10位
        for row in range(3):
            for column in range(3):
                # 注意abacus的结构文件以bohr为长度单位，数据要先除以转换系数，再除以晶格系数
                template += f"    {(self.lattice_vectors[row][column] / Bohr2Angstrom / self.lattice_constant):.10f}"
            template += "\n"
        template += "\n"

        # 坐标种类部分
        template += "ATOMIC_POSITIONS\n"
        template += f"{self.coordinate_type}\n"
        template += "\n"

        # 原子信息部分
        for element in self.atomic_species:
            # 第1行是元素类型
            template += f"{element}\n"

            # 第2行是极化强度
            template += f"{self.magnetism[element]}\n"

            # 第3行是对应元素的原子数N
            atom_num = len(self.atomic_positions[element])
            template += f"{atom_num}\n"

            # 第4行开始是原子坐标。逐个元素打印这个矩阵，控制输出精度为小数点后10位
            for nn in range(atom_num):
                for xx in range(3):
                    template += f"    {self.atomic_positions[element][nn][xx]:.10f}"
                # 每行末尾加上允许原子位移的参数
                template += f" m 1 1 1\n"
            template += "\n"

        # 将模板写入到文件
        with open(output_file, 'w') as f:
            f.write(template)

        return

    @classmethod
    def from_lammps(cls, conf_file, type_map: dict):
        """从LAMMPS结构文件构造对象。需要提供模型训练的type_map"""
        instance = cls()

        xlo, xhi = 0.0, 0.0
        ylo, yhi = 0.0, 0.0
        zlo, zhi = 0.0, 0.0
        xy, xz, yz = 0.0, 0.0, 0.0

        # 建一个list，元素为tuple格式，存储元素名及对应原子坐标信息：(element，[x, y, z])
        atom_coords = []

        with open(conf_file, 'r') as f:
            current_section = None
            for line in f:
                # 去除行首尾的空格和换行符
                line = line.strip()
                # 跳过空行
                if not line:
                    continue

                # 解析晶格矢量部分（模拟盒子）
                if "xlo xhi" in line:
                    xlo, xhi = map(float, line.split()[:2])
                elif "ylo yhi" in line:
                    ylo, yhi = map(float, line.split()[:2])
                elif "zlo zhi" in line:
                    zlo, zhi = map(float, line.split()[:2])
                elif "xy xz yz" in line:
                    xy, xz, yz = map(float, line.split()[:3])

                # 标记原子type编号和对应原子质量部分（如果有的话），在接下来开始处理
                elif line.startswith("Masses"):
                    current_section = "Masses"

                # 标记原子id、type编号、原子坐标部分，在接下来开始处理
                elif line.startswith("Atoms"):
                    current_section = "Atoms"

                # 不需要存储原子速度信息，将current_section设为None跳过这部分的解析
                elif line.startswith("Velocities"):
                    current_section = "None"

                else:
                    # 开始解析原子type编号和对应原子质量部分（如果有的话）
                    if current_section == "Masses":
                        parts = line.split()
                        # 第一项是type编号
                        type_num = int(parts[0])
                        # 第二项是相对原子质量
                        mass_value = float(parts[1])

                        # 遍历参数type_map字典，寻找编号对应的元素名
                        for element_key in type_map:
                            # type_map字典里编号从0开始，读取的文件里的type编号要减1
                            if type_map[element_key] == type_num - 1:
                                # 将对应的原子质量存储到成员变量masses字典里
                                instance.masses[element_key] = mass_value
                                break

                    # 解析原子id、type编号、原子坐标部分
                    elif current_section == "Atoms":
                        parts = line.split()
                        # type编号在第二项
                        type_num = int(parts[1])
                        # type编号后紧接着三项分别是x, y, z坐标
                        x, y, z = map(float, parts[2:5])

                        # 遍历传入参数type_map字典，寻找编号对应的元素名
                        for element_key in type_map:
                            # type_map字典里编号从0开始，读取的文件里的type编号要减1
                            if type_map[element_key] == type_num - 1:
                                # 记录元素名和对应坐标信息，以tuple格式存储
                                atom_coords.append((element_key, [x, y, z]))
                                break

                    # 既不是空行又没有关键字的部分直接跳过
                    else:
                        continue

        # 根据lammps文档：https://docs.lammps.org/read_data.html，计算晶格矢量
        instance.lattice_vectors = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [xy, yhi - ylo, 0.0],
                [xz, yz, zhi - zlo]
            ]
        )

        # lammps结构文件格式默认为Cartesian坐标。单位需要在in文件中设置为metal以对应Angstrom
        instance.coordinate_type = "Cartesian"

        # 把atom_coords变量里存储的信息转换成字典格式存到成员atomic_positions里
        for element, coord in atom_coords:
            # 检查当前元素是否在字典中已存在key
            if element in instance.atomic_positions:
                # 如果存在，将当前coord追加到list里
                instance.atomic_positions[element].append(coord)
            else:
                # 如果不存在，创建一个新list，并把coord作为第一个元素
                instance.atomic_positions[element] = [coord]

        for element_key in instance.atomic_positions:
            # 将坐标数据从list转为numpy矩阵格式
            instance.atomic_positions[element_key] = np.array(instance.atomic_positions[element_key])

            # 将结构文件中出现的元素存储到成员atomic_species里
            instance.atomic_species.append(element_key)

        # 如果成员masses为空，根据type_map中的信息补齐
        if len(instance.masses) == 0:
            for element_key in type_map:
                instance.masses[element_key] = get_mass(element_key)

        return instance

    def to_lammps(self, output_file, type_map: dict):
        """输出LAMMPS格式结构文件"""
        # 构造一个输出模板
        template = ""

        # 根据成员atomic_positions里各元素坐标矩阵的行数，统计系统总原子数
        total_atom_num: int = 0
        for element in self.atomic_positions:
            total_atom_num += len(self.atomic_positions[element])

        # 第一部分：总原子数与原子种类数。原子种类数要与训练模型一致，根据参数type_map获得
        template += "\n"
        template += f"{total_atom_num} atoms\n"
        template += f"{len(type_map)} atom types\n"
        template += "\n"

        # 第二部分：模拟盒子信息，LAMMPS格式的晶格矢量记法。控制输出小数点后10位
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        xhi = self.lattice_vectors[0][0]
        yhi = self.lattice_vectors[1][1]
        zhi = self.lattice_vectors[2][2]
        xy = self.lattice_vectors[1][0]
        xz = self.lattice_vectors[2][0]
        yz = self.lattice_vectors[2][1]
        template += f"{xlo:.10f} {xhi:.10f} xlo xhi\n"
        template += f"{ylo:.10f} {yhi:.10f} ylo yhi\n"
        template += f"{zlo:.10f} {zhi:.10f} zlo zhi\n"
        template += f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n"
        template += "\n"

        # 第三部分：质量信息，原子type编号、相对原子质量。控制输出小数点后4位
        template += "Masses\n"
        template += "\n"
        for element in self.masses:
            # LAMMPS中原子type编号从1开始，索引值需要加1
            type_num: int = type_map[element] + 1
            template += f"{type_num} {self.masses[element]:.4f}\n"
        template += "\n"

        # 第四部分：原子id、原子type编号、xyz的cartesian坐标。坐标数值控制输出小数点后10位
        template += "Atoms # atomic\n"
        template += "\n"
        id_num: int = 1
        for element in self.atomic_positions:
            # LAMMPS中原子type编号从1开始，索引值需要加1
            type_num: int = type_map[element] + 1
            for coord in self.atomic_positions[element]:
                x, y, z = coord[0], coord[1], coord[2]
                template += f"{id_num} {type_num} {x:.10f} {y:.10f} {z:.10f}\n"
                id_num += 1
        template += "\n"

        # 将模板写入到文件
        with open(output_file, 'w') as f:
            f.write(template)

        return

    def apply_deform(self, deform):
        """将变形张量作用于结构，得到新晶格矢量和新的笛卡尔坐标，分数坐标不变"""
        # abacus结构文件格式下，新晶格矢量 = 转置(变形张量 点乘 转置(旧晶格矢量)) = 旧晶格矢量 点乘 转置(变形张量)
        new_lattice_vectors = np.dot(self.lattice_vectors, deform.deform_matrix.transpose())
        self.lattice_vectors = new_lattice_vectors

        # 如果是笛卡尔坐标，所有坐标也要进行变换
        if self.coordinate_type == "Cartesian":
            for element in self.atomic_species:
                new_positions = np.dot(self.atomic_positions[element], deform.deform_matrix.transpose())
                self.atomic_positions[element] = new_positions

        return self

    def get_volume(self):
        """根据晶格矢量，计算结构单胞的体积，单位为Angstrom^3"""
        a = self.lattice_vectors[0]
        b = self.lattice_vectors[1]
        c = self.lattice_vectors[2]
        return np.dot(a, np.cross(b, c))
