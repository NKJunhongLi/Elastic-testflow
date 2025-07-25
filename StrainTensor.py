import numpy as np
import itertools
from math import sqrt
from abc import ABC, abstractmethod


class DeformationTensor:
    """变形梯度张量，以3×3的numpy矩阵存储，构造为上三角矩阵"""
    def __init__(self):
        """DeformationTensor类构造函数。确定成员变量"""
        # 3×3矩阵形式，初始化为单位矩阵
        self.deform_matrix = np.eye(3)

    @classmethod
    def from_GL_index(cls, index, amount):
        """根据指标和对应Green_Lagrange应变分量的值构造变形张量。参数amount的输入为应变分量的值"""
        instance = cls()

        # 为了后面的读取，强制要求输入的index为tuple格式
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("To use function DeformationTensor.from_GL_index(), index must be a 2-tuple")

        if index in {(0, 0), (1, 1), (2, 2)}:
            instance.deform_matrix[index] = sqrt(1 + 2 * amount)

        elif index in {(0, 1), (0, 2), (1, 2)}:
            instance.deform_matrix[index] = 2 * amount

            # 操作对应的对角元素：(1,1)或(2,2)
            instance.deform_matrix[(index[1], index[1])] = sqrt(1 - 4 * amount**2)

        elif index in {(1, 0), (2, 0), (2, 1)}:
            instance.deform_matrix[index] = 2 * amount

            # 操作对应的对角元素：(1,1)或(2,2)
            instance.deform_matrix[(index[0], index[0])] = sqrt(1 - 4 * amount**2)

            # 进行转置操作，使其为上三角矩阵
            instance.deform_matrix.transpose()

        else:
            raise ValueError("The index is out of range! Green-Lagrange strain tensor is a 3 * 3 matrix!")

        return instance

    @classmethod
    def from_GL_strain(cls, strain: 'GreenLagrangeStrain'):
        """根据Green_lagrange应变张量计算得到变形张量"""
        instance = cls()

        # 右柯西-格林应变张量 = 转置(变形张量) 点乘 变形张量 = 2*应变张量 - 单位张量
        D = 2 * strain.matrix - np.eye(3)

        # np.linalg.cholesky分解输出下三角矩阵，因此需要一次转置操作
        instance.deform_matrix = np.linalg.cholesky(D).transpose()

        return instance

    @classmethod
    def from_kittel_index(cls, index, amount):
        """根据指标和对应Kittel规范应变分量的值构造变形张量。参数amount的输入为应变分量的值"""
        instance = cls()

        # 为了后面的读取，强制要求输入的index为tuple格式
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("To use function DeformationTensor.from_kittel_index(), index must be a 2-tuple")

        if index in {(0, 0), (1, 1), (2, 2)}:
            instance.deform_matrix[index] += amount

        elif index in {(0, 1), (0, 2), (1, 2)}:
            instance.deform_matrix[index] = amount

        elif index in {(1, 0), (2, 0), (2, 1)}:
            instance.deform_matrix[index] = amount

            # 进行转置操作，使其为上三角矩阵
            instance.deform_matrix.transpose()

        else:
            raise ValueError("The index is out of range! Strain tensor is a 3 * 3 matrix!")

        return instance

    @classmethod
    def from_kittel_strain(cls, strain: 'KittelStrain'):
        """根据Kittel规范应变张量计算变形张量"""
        instance = cls()

        # Kittel规范下，已知应变张量构造变形张量非常简单：变形张量 = 单位张量 + 应变张量上三角部分
        for row in range(3):
            for column in range(3):
                if column >= row:
                    instance.deform_matrix[row][column] += strain.matrix[row][column]

        return instance


class StrainTensor(ABC):
    """
    抽象基类，声明应力张量类的成员变量和成员函数\n
    成员变量有两个：3×3 numpy矩阵类型，用来存储应力张量。6×1 numpy矩阵类型的列向量，用来存储voigt notation格式的应力张量\n
    抽象函数有一个：from_deform()，用于根据给定变形张量计算应变张量\n
    """
    def __init__(self):
        # 3×3 numpy矩阵，存储矩阵表示形式的应力张量
        self.matrix = np.zeros((3, 3))

        # 6×1 numpy矩阵，存储voigt notation格式的应力张量
        self.voigt = np.zeros((6, 1))

    @classmethod
    def from_index(cls, index, amount):
        """根据指标和对应分量的值构造应变张量"""
        instance = cls()

        # 为了后面的读取，强制要求输入的index为tuple格式
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("To use function StrainTensor.from_index(), index must be a 2-tuple")

        # 使用itertools.permutations对输入的坐标进行全排列，然后把amount赋值到对应分量。保证结果是一个对称矩阵
        for ii in itertools.permutations(index):
            instance.matrix[ii] = amount

        # 对应应变张量的分量，给Voigt notation下的分量赋值
        instance.set_voigt()

        return instance

    @classmethod
    @abstractmethod
    def from_deform(cls, deform: DeformationTensor):
        pass

    @abstractmethod
    def to_deform(self) -> DeformationTensor:
        pass

    def set_voigt(self):
        """对应应力张量的分量，给Voigt notation下的分量赋值"""
        self.voigt[0] = self.matrix[(0, 0)]
        self.voigt[1] = self.matrix[(1, 1)]
        self.voigt[2] = self.matrix[(2, 2)]
        self.voigt[3] = self.matrix[(1, 2)]
        self.voigt[4] = self.matrix[(0, 2)]
        self.voigt[5] = self.matrix[(0, 1)]
        return


class GreenLagrangeStrain(StrainTensor):
    """
    存储Green-Lagrange应变张量的类，继承自抽象基类StrainTensor。有两个类成员变量：\n
    格林-拉格朗日应变张量，以3×3的numpy矩阵存储\n
    Voigt notation下的应变张量，以6×1的numpy矩阵存储
    """
    @classmethod
    def from_deform(cls, deform: DeformationTensor):
        """根据变形张量计算获得Green-Lagrange应变张量"""
        instance = cls()

        # 右柯西-格林应变张量 = 转置(变形张量) 点乘 变形张量
        D = np.dot(deform.deform_matrix.transpose(), deform.deform_matrix)

        # 格林-拉格朗日应变张量 = (1/2) * (右柯西-格林应变张量 - 单位张量)
        instance.matrix = (D - np.eye(3)) / 2

        # 对应应变张量的分量，给Voigt notation下的分量赋值
        instance.set_voigt()

        return instance

    def to_deform(self) -> DeformationTensor:
        """根据存储的应变张量信息，返回一个对应的变形张量"""
        return DeformationTensor.from_GL_strain(self)


class KittelStrain(StrainTensor):
    """
    存储符合Kittel《固体物理导论》书中定义规范下的应变张量，继承自抽象基类StrainTensor。有两个类成员：
    Kittel规范的应变张量，以3*3的numpy矩阵存储\n
    Voigt notation下的Kittel规范应变张量，以6*1的numpy矩阵存储
    """
    @classmethod
    def from_deform(cls, deform: DeformationTensor):
        """根据变形张量计算获得Kittel规范下应变张量"""
        instance = cls()

        # 对角元素，F_ii = 1 + E_ii; E_ii = F_ii - 1
        for index in [(0, 0), (1, 1), (2, 2)]:
            instance.matrix[index] = deform.deform_matrix[index] - 1

        # 非对角元素：E_ij = F_ij + F_ji。由于代码限定了变形矩阵为上三角型，因此 E_ij = E_ji = F_ij, j>i
        for index in [(0, 1), (0, 2), (1, 2)]:
            instance.matrix[index] = deform.deform_matrix[index]
            # 给对称项赋相同值，保证它为对称矩阵
            instance.matrix[tuple(reversed(index))] = instance.matrix[index]

        # 对应应变张量的分量，给Voigt notation下的分量赋值
        instance.set_voigt()

        return instance

    def to_deform(self) -> DeformationTensor:
        """根据存储的应变张量信息，返回一个对应的变形张量"""
        return DeformationTensor.from_kittel_strain(self)


def make_norm_deforms(norm_deform=0.01):
    """根据配置输入的正应变强度，输出存有12个DeformationTensor的list"""
    make_list = [-1.0 * norm_deform, -0.5 * norm_deform, 0.5 * norm_deform, 1.0 * norm_deform]
    deform_list = []
    for index in [(0, 0), (1, 1), (2, 2)]:
        for deform in make_list:
            '''
            # 按照Green-Lagrange应变张量的规范构造
            deform_list.append(DeformationTensor.from_GL_index(index, deform))
            '''

            # 按照Kittel的规范构造
            deform_list.append(DeformationTensor.from_kittel_index(index, deform))

    return deform_list


def make_shear_deforms(shear_deform=0.01):
    """根据配置输入的剪切应变强度，输出存有12个DeformationTensor的list"""
    make_list = [-1.0 * shear_deform, -0.5 * shear_deform, 0.5 * shear_deform, 1.0 * shear_deform]
    deform_list = []
    for index in [(1, 2), (0, 2), (0, 1)]:
        for deform in make_list:
            '''
            # 按照Green-Lagrange应变张量的规范构造
            deform_list.append(DeformationTensor.from_GL_index(index, deform))
            '''

            # 按照Kittel的规范构造
            deform_list.append(DeformationTensor.from_kittel_index(index, deform))

    return deform_list


def make_small_deforms(small_deform=1e-4):
    """构造一个字典，存储正负共12个小应变分量对应的变形张量，用于中心差分算法。"""
    '''
    # 按照Green-Lagrange应变张量的规范构造
    small_deform_dict = {
        "xx+": DeformationTensor.from_GL_index((0, 0), small_deform),
        "yy+": DeformationTensor.from_GL_index((1, 1), small_deform),
        "zz+": DeformationTensor.from_GL_index((2, 2), small_deform),
        "yz+": DeformationTensor.from_GL_index((1, 2), small_deform),
        "xz+": DeformationTensor.from_GL_index((0, 2), small_deform),
        "xy+": DeformationTensor.from_GL_index((0, 1), small_deform),
        "xx-": DeformationTensor.from_GL_index((0, 0), -1.0 * small_deform),
        "yy-": DeformationTensor.from_GL_index((1, 1), -1.0 * small_deform),
        "zz-": DeformationTensor.from_GL_index((2, 2), -1.0 * small_deform),
        "yz-": DeformationTensor.from_GL_index((1, 2), -1.0 * small_deform),
        "xz-": DeformationTensor.from_GL_index((0, 2), -1.0 * small_deform),
        "xy-": DeformationTensor.from_GL_index((0, 1), -1.0 * small_deform)
    }
    '''

    # 按照Kittel的规范构造
    small_deform_dict = {
        "xx+": DeformationTensor.from_kittel_index((0, 0), small_deform),
        "yy+": DeformationTensor.from_kittel_index((1, 1), small_deform),
        "zz+": DeformationTensor.from_kittel_index((2, 2), small_deform),
        "yz+": DeformationTensor.from_kittel_index((1, 2), small_deform),
        "xz+": DeformationTensor.from_kittel_index((0, 2), small_deform),
        "xy+": DeformationTensor.from_kittel_index((0, 1), small_deform),
        "xx-": DeformationTensor.from_kittel_index((0, 0), -1.0 * small_deform),
        "yy-": DeformationTensor.from_kittel_index((1, 1), -1.0 * small_deform),
        "zz-": DeformationTensor.from_kittel_index((2, 2), -1.0 * small_deform),
        "yz-": DeformationTensor.from_kittel_index((1, 2), -1.0 * small_deform),
        "xz-": DeformationTensor.from_kittel_index((0, 2), -1.0 * small_deform),
        "xy-": DeformationTensor.from_kittel_index((0, 1), -1.0 * small_deform)
    }

    return small_deform_dict
