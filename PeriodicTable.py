"""包含存储所有元素相对原子质量的list，以及获得原子质量、获取元素名、获取原子序数的函数。数据来源IUPAC 2022 standard atomic weights"""

# 以list格式存储所有元素名和相对原子质量。list中以tuple格式存储，第1项为str格式元素名，第2项为float格式相对原子质量
periodic_table = [
    ("H", 1.0080),
    ("He", 4.0026),
    ("Li", 6.94),
    ("Be", 9.0122),
    ("B", 10.81),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Ne", 20.180),
    ("Na", 22.990),
    ("Mg", 24.305),
    ("Al", 26.982),
    ("Si", 28.0855),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("Ar", 39.95),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Sc", 44.956),
    ("Ti", 47.867),
    ("V", 50.942),
    ("Cr", 51.996),
    ("Mn", 54.938),
    ("Fe", 55.845),
    ("Co", 58.933),
    ("Ni", 58.693),
    ("Cu", 63.546),
    ("Zn", 65.38),
    ("Ga", 69.723),
    ("Ge", 72.630),
    ("As", 74.922),
    ("Se", 78.971),
    ("Br", 79.904),
    ("Kr", 83.798),
    ("Rb", 85.468),
    ("Sr", 87.62),
    ("Y", 88.906),
    ("Zr", 91.224),
    ("Nb", 92.906),
    ("Mo", 95.95),
    ("Tc", 97.0),
    ("Ru", 101.07),
    ("Rh", 102.91),
    ("Pd", 106.42),
    ("Ag", 107.87),
    ("Cd", 112.41),
    ("In", 114.82),
    ("Sn", 118.71),
    ("Sb", 121.76),
    ("Te", 127.60),
    ("I", 126.90),
    ("Xe", 131.29),
    ("Cs", 132.91),
    ("Ba", 137.33),
    ("La", 138.91),
    ("Ce", 140.12),
    ("Pr", 140.91),
    ("Nd", 144.24),
    ("Pm", 145.0),
    ("Sm", 150.36),
    ("Eu", 151.96),
    ("Gd", 157.25),
    ("Tb", 158.93),
    ("Dy", 162.50),
    ("Ho", 164.93),
    ("Er", 167.26),
    ("Tm", 168.93),
    ("Yb", 173.05),
    ("Lu", 174.97),
    ("Hf", 178.49),
    ("Ta", 180.95),
    ("W", 183.84),
    ("Re", 186.21),
    ("Os", 190.23),
    ("Ir", 192.22),
    ("Pt", 195.08),
    ("Au", 196.97),
    ("Hg", 200.59),
    ("Tl", 204.38),
    ("Pb", 207.2),
    ("Bi", 208.98),
    ("Po", 209.0),
    ("At", 210.0),
    ("Rn", 222.0),
    ("Fr", 223.0),
    ("Ra", 226.0),
    ("Ac", 227.0),
    ("Th", 232.04),
    ("Pa", 231.04),
    ("U", 238.03),
    ("Np", 237.0),
    ("Pu", 244.0),
    ("Am", 243.0),
    ("Cm", 247.0),
    ("Bk", 247.0),
    ("Cf", 251.0),
    ("Es", 252.0),
    ("Fm", 257.0),
    ("Md", 258.0),
    ("No", 259.0),
    ("Lr", 262.0),
    ("Rf", 267.0),
    ("Db", 268.0),
    ("Sg", 269.0),
    ("Bh", 270.0),
    ("Hs", 269.0),
    ("Mt", 277.0),
    ("Ds", 281.0),
    ("Rg", 282.0),
    ("Cn", 285.0),
    ("Nh", 286.0),
    ("Fl", 290.0),
    ("Mc", 290.0),
    ("Lv", 293.0),
    ("Ts", 294.0),
    ("Og", 294.0)
]


def get_mass(input_arg) -> float:
    """
    根据输入参数获得对应的相对原子质量，自动识别输入参数类型\n
    如果输入int型，识别为原子序数，返回对应原子序数的元素的相对原子质量\n
    如果输入str型，识别为元素名，返回对应元素名的相对原子质量
    """
    # 处理int型输入参数，识别为原子序数
    if isinstance(input_arg, int):
        print(f"Searching atomic number {input_arg}\n")

        # 如果输入的原子序数不合理，主动报错
        if input_arg <= 0 or input_arg > len(periodic_table):
            raise ValueError(f"Error! Atomic number should be in range from 1 to {len(periodic_table)}")

        # list下标从0开始，索引要减1
        element = periodic_table[input_arg - 1]
        print(f"Searching result: Atom symbol {element[0]}, standard atomic weight {element[1]}\n")
        return element[1]

    # 处理str型输入参数，识别为元素符号
    elif isinstance(input_arg, str):
        print(f"Searching atom symbol {input_arg}\n")

        # 遍历元素周期表
        for ii in range(len(periodic_table)):
            element = periodic_table[ii]
            # 原子序数 = 下标 + 1
            atomic_number = ii + 1
            if element[0] == input_arg:
                print(f"Searching result: Atomic number {atomic_number}, standard atomic weight {element[1]}\n")
                return element[1]

        # 如果遍历完没找到对应元素，主动报错
        raise ValueError(f"Error! Atom symbol {input_arg} not found!")

    # 对于其它类型的参数输入，直接报错
    else:
        raise ValueError("Error! Please input atomic number(int) or atom symbol(str) as argument")


def get_atomic_number(input_arg) -> int:
    """
        根据输入参数获得对应的原子序数，自动识别输入参数类型\n
        如果输入float型，识别为相对原子质量，搜索返回数值最接近的元素的原子序数。如果有多个匹配，返回原子序数最大的那个\n
        如果输入str型，识别为元素名，返回对应元素名的原子序数
    """
    # 处理float型参数，识别为相对原子质量
    if isinstance(input_arg, float):
        print(f"Searching standard atomic weight {input_arg}\n")

        # 设置最大误差为1.0，不接受绝对误差大于1.0的搜索结果
        err = 1.0

        # 初始化输出
        result: int = 0

        # 遍历元素周期表
        for ii in range(len(periodic_table)):
            element = periodic_table[ii]
            # 原子序数 = 下标 + 1
            atomic_number = ii + 1

            # 如果找到相对原子质量的绝对误差更小的对应元素，更新输出
            current_err = abs(element[1] - input_arg)
            if current_err < err:
                err = current_err
                result = atomic_number
                print(
                    f"Probable result: Atomic number {atomic_number}, "
                    f"atom symbol {element[0]}, "
                    f"standard atomic mass {element[1]}\n"
                )

        # 如果遍历完整个元素周期表找不到接近输入参数的结果，主动报错
        if result == 0:
            raise ValueError(f"Cannot found one result approximately equal to {input_arg}\n")

        print(f"Return atomic number {result} as result\n")
        return result

    # 处理str型输入参数，识别为元素符号
    elif isinstance(input_arg, str):
        print(f"Searching atom symbol {input_arg}\n")

        # 遍历元素周期表
        for ii in range(len(periodic_table)):
            element = periodic_table[ii]
            # 原子序数 = 下标 + 1
            atomic_number = ii + 1
            if element[0] == input_arg:
                print(f"Searching result: Atomic number {atomic_number}, standard atomic weight {element[1]}\n")
                return atomic_number

        # 如果遍历完没找到对应元素，主动报错
        raise ValueError(f"Error! Atom symbol {input_arg} not found!")

    # 对于其它类型的参数输入，直接报错
    else:
        raise ValueError("Error! Please input standard atomic weight(float) or atom symbol(str) as argument")


def get_atom_symbol(input_arg) -> str:
    """
        根据输入参数获得对应的元素符号，自动识别输入参数类型\n
        如果输入float型，识别为相对原子质量，搜索返回数值最接近的元素符号。如果有多个匹配，返回原子序数最大的那个\n
        如果输入int型，识别为原子序数，返回对应原子序数的元素名
    """
    # 处理float型参数，识别为相对原子质量
    if isinstance(input_arg, float):
        print(f"Searching standard atomic weight {input_arg}\n")

        # 设置最大误差为1.0，不接受绝对误差大于1.0的搜索结果
        err = 1.0

        # 初始化输出
        result: str = ""

        # 遍历元素周期表
        for ii in range(len(periodic_table)):
            element = periodic_table[ii]
            # 原子序数 = 下标 + 1
            atomic_number = ii + 1

            # 如果找到相对原子质量的绝对误差更小的对应元素，更新输出
            current_err = abs(element[1] - input_arg)
            if current_err < err:
                err = current_err
                result = element[0]
                print(
                    f"Probable result: Atomic number {atomic_number}, "
                    f"atom symbol {element[0]}, "
                    f"standard atomic mass {element[1]}\n"
                )

        # 如果遍历完整个元素周期表找不到接近输入参数的结果，主动报错
        if result == "":
            raise ValueError(f"Cannot found one result approximately equal to {input_arg}\n")

        print(f"Return atom symbol {result} as result\n")
        return result

    # 处理int型输入参数，识别为原子序数
    elif isinstance(input_arg, int):
        print(f"Searching atomic number {input_arg}\n")

        # 如果输入的原子序数不合理，主动报错
        if input_arg <= 0 or input_arg > len(periodic_table):
            raise ValueError(f"Error! Atomic number should be in range from 1 to {len(periodic_table)}")

        # list下标从0开始，索引要减1
        element = periodic_table[input_arg - 1]
        print(f"Searching result: Atom symbol {element[0]}, standard atomic weight {element[1]}\n")
        return element[0]

    # 对于其它类型的参数输入，直接报错
    else:
        raise ValueError("Error! Please input atomic number(int) or standard atomic weight(float) as argument")
