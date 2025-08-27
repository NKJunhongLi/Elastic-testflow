# Elastic-testflow
A python program for calculating elastic tensor and modulus, connecting with DFT software ABACUS or MD software LAMMPS. Energy finite differential method for calculating stress are also provided.

# Stress-Strain线性拟合方法计算弹性常数矩阵

## 程序简介
计算弹性常数矩阵的方法有Energy-Strain拟合的方法，也有Stress-Strain拟合的方法。Energy-Strain方法是选定基准能量后计算不同变胞结构的能量差值作为弹性能，通过弹性能-应变的关系逐个分量拟合得到弹性常数矩阵。Stress-Strain方法则是直接计算各个变胞结构的应力，通过应力-应变的线性矩阵方程直接得到弹性常数矩阵。

本程序使用Stress-Strain方法，与dpgen autotest和materials project pymatgen程序包在流程上保持一致。不同点在于，为更好适配ABACUS软件，本程序的应力张量规范使用Kittel书《Introduction to Solid States physics》中的定义；pymatgen程序包使用Green-Lagrange的定义。

## 完整计算流程

### 结构优化
这一步未在程序中实现，输入文件的准备和提交计算需要单独手动完成。

准备好初始结构文件，初始结构文件可以在Materials Project官网上下载POSCAR或.cif格式的文件，然后使用atomkit转换成ABACUS的STRU格式或LAMMPS的.lmp格式。DFT方法使用ABACUS软件，MD方法使用LAMMPS软件，进行允许晶格变形的结构优化。

#### 使用ABACUS
以ABACUS计算碳原子金刚石结构为例，STRU格式文件如下

```
ATOMIC_SPECIES
C   12.011  C.upf

NUMERICAL_ORBITAL
C_gga_7au_100Ry_2s2p1d.orb

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
3.573710000000  0.000000000000  0.000000000000
0.000000000000  3.573710000000  0.000000000000
0.000000000000  0.000000000000  3.573710000000

ATOMIC_POSITIONS
Direct

C 
0.000
8
0.250000000000  0.750000000000  0.250000000000  1  1  1  mag  0.0
0.000000000000  0.000000000000  0.500000000000  1  1  1  mag  0.0
0.250000000000  0.250000000000  0.750000000000  1  1  1  mag  0.0
0.000000000000  0.500000000000  0.000000000000  1  1  1  mag  0.0
0.750000000000  0.750000000000  0.750000000000  1  1  1  mag  0.0
0.500000000000  0.000000000000  0.000000000000  1  1  1  mag  0.0
0.750000000000  0.250000000000  0.250000000000  1  1  1  mag  0.0
0.500000000000  0.500000000000  0.500000000000  1  1  1  mag  0.0
```
准备输入文件INPUT，参考示例如下。注意calculation要设置为cell-relax，且cal_force和cal_stress要打开。
```
INPUT_PARAMETERS
#Parameters (General)
suffix         ABACUS
pseudo_dir     /home/lijh/abacus/selected-Dojo/Pseudopotential/
orbital_dir    /home/lijh/abacus/selected-Dojo/selected_Orbs/
nspin          1    # 1 or 2 or 4. The number of spin components of wave functions. 1 for Spin degeneracy, 2 for Collinear spin polarized, 4 for SOC. Default 1
symmetry       1    # 0 or 1. 1 for open, default 1.
esolver_type   ksdft    # ksdft, ofdft, sdft, tddft, etc. Choose the energy solver. Default ksdft.
ks_solver      genelpa    # Choose the diagonalization method for the Hamiltonian matrix expanded in a certain basis set. Default cg for pw. Default genelpa for lcao.

#Parameters (Methods)
ecutwfc        100
scf_thr        1e-8
scf_nmax       300
relax_nmax     200
relax_method   cg    # Conjugate gradient algorithm.
force_thr_ev   0.01
stress_thr     0.5
basis_type     lcao
calculation    cell-relax    # structure relaxation calculations
dft_functional PBE

#Parameters (Smearing)
smearing_method gaussian    # recommend to set fixed for non-conductors, gaussian for semi-conductors, mp for metal. Default gaussian
smearing_sigma  0.010    # Rydberg, energy range for smearing, default 0.015.

#Parameters (charge mixing and k points spacing)
kspacing       0.1
mixing_beta     0.5    # The formula for charge mixing be written as rho_new = rho_old + beta * rho_update. Default 0.8 for nspin=1, 0.4 for nspin=2 and nspin=4

#Parameters (File)
cal_force      1
cal_stress     1
```
INPUT里设置了`kspacing`，所以可以不用准备KPT文件；如果不设置这一项则需要单独准备KPT文件。示例如下
```
K_POINTS
0
Gamma
10 10 10 0 0 0
```
准备好后运行计算任务。计算结果输出在`OUT.ABACUS/`目录里，优化后的结构文件为`STRU_ION_D`，输出文件为`running_cell-relax.log`。

#### 使用 LAMMPS + DP势函数模型
以LAMMPS计算t相二氧化铪HfO2为例，结构文件示例如下，命名为`conf.lmp`
```

6 atoms
5 atom types
   0.0000000000    3.5933280000 xlo xhi
   0.0000000000    3.5933280000 ylo yhi
   0.0000000000    5.2246730000 zlo zhi
   0.0000000000    0.0000000000    0.0000000000 xy xz yz

Masses

1 178.4900
2 91.2240
3 88.9060
4 26.9820
5 15.9990

Atoms # atomic

     1      1    1.7966640000    1.7966640000    2.6123365000
     2      1    0.0000000000    0.0000000000    0.0000000000
     3      5    0.0000000000    1.7966640000    3.6327569343
     4      5    1.7966640000    0.0000000000    4.2042525657
     5      5    1.7966640000    0.0000000000    1.5919160657
     6      5    0.0000000000    1.7966640000    1.0204204343
```
注意，当使用DP模型时，要把训练时的type_map元素序号和相对原子质量写入结构文件里。如果不这么做，则需要在输入文件in.lammps里逐个设置元素序号及其对应质量。元素需要一定要和训练DP模型时一致。  
输入文件`in.lammps`示例如下
```
clear
units    metal
dimension    3
boundary    p p p
box    tilt large

read_data    conf.lmp

neigh_modify    every 1 delay 0 check no

pair_style    deepmd    graph.pb
pair_coeff * *

compute    mype all pe
thermo    100
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz lx ly lz vol c_mype
dump    1 all custom 100 dump.relax id type xs ys zs fx fy fz

write_data    CONTCAR.lmp

min_style       cg
fix             1 all box/relax iso 0.0 
minimize        0.000000e+00 1.000000e-10 5000 500000
fix             1 all box/relax aniso 0.0 
minimize        0.000000e+00 1.000000e-10 5000 500000
fix             1 all box/relax tri 0.0 
minimize        0.000000e+00 1.000000e-10 5000 500000

variable    E equal "c_mype"
variable    Pxx equal pxx
variable    Pyy equal pyy
variable    Pzz equal pzz
variable    Pyz equal pyz
variable    Pxz equal pxz
variable    Pxy equal pxy

print "All done"
print "Final energy = ${E}"
print "Final Stress (xx yy zz yz xz xy) = ${Pxx} ${Pyy} ${Pzz} ${Pyz} ${Pxz} ${Pxy}"

```
这是经典的使用LAMMPS软件进行cell-relax结构优化的流程，经历各向同性体积弛豫-各向异性晶格弛豫-三斜弛豫3个步骤达到高精度的力学平衡结构。此外，设置write_data可以将优化后的结构文件输出。  
要注意，此示例文件中
```
compute    mype all pe

variable    E equal "c_mype"
variable    Pxx equal pxx
variable    Pyy equal pyy
variable    Pzz equal pzz
variable    Pyz equal pyz
variable    Pxz equal pxz
variable    Pxy equal pxy

print "All done"
print "Final energy = ${E}"
print "Final Stress (xx yy zz yz xz xy) = ${Pxx} ${Pyy} ${Pzz} ${Pyz} ${Pxz} ${Pxy}"
```
这段指令为必须，这段指令是用来在输出文件log.lammps中打印能量和应力信息。本程序读取LAMMPS日志是按照此输入输出来写的，如果不加这段指令，会导致程序读取log.lammps信息报错。  
准备好后执行计算任务，输出文件在与输入文件相同目录。`log.lammps`是屏幕输入输出信息，`dump.relax`是原子步信息，CONTCAR.lmp是优化后的结构文件。

### 生成变胞结构
在获得充分弛豫、完成结构优化的结构文件后，我们需要对它施加不同大小的应变，生成一系列变胞结构文件。

在Kittel的规范中，对于晶格变形的矩阵关系
$$
\left[\begin{array}{ccc}
a_{1x}^{new} & a_{2x}^{new} & a_{3x}^{new}\\
a_{1y}^{new} & a_{2y}^{new} & a_{3y}^{new}\\
a_{1z}^{new} & a_{2z}^{new} & a_{3z}^{new}
\end{array}\right]=\left[\begin{array}{ccc}
1+\gamma_{xx} & \gamma_{xy} & \gamma_{xz}\\
\gamma_{yx} & 1+\gamma_{yy} & \gamma_{yz}\\
\gamma_{zx} & \gamma_{zy} & 1+\gamma_{zz}
\end{array}\right]\left[\begin{array}{ccc}
a_{1x} & a_{2x} & a_{3x}\\
a_{1y} & a_{2y} & a_{3y}\\
a_{1z} & a_{2z} & a_{3z}
\end{array}\right]
$$
应变定义为
$$
\boldsymbol{e}\doteq\left[\begin{array}{ccc}
\gamma_{xx} & \left(\gamma_{xy}+\gamma_{yx}\right) & \left(\gamma_{xz}+\gamma_{zx}\right)\\
\left(\gamma_{xy}+\gamma_{yx}\right) & \gamma_{yy} & \left(\gamma_{yz}+\gamma_{zy}\right)\\
\left(\gamma_{xz}+\gamma_{zx}\right) & \left(\gamma_{yz}+\gamma_{zy}\right) & \gamma_{zz}
\end{array}\right]
$$
本程序代码中，我们将变形矩阵限定为上三角矩阵，则应变矩阵$F$和变形矩阵$e$的关系可以写为
$$
\boldsymbol{F}=\left[\begin{array}{ccc}
1+e_{xx} & e_{xy} & e_{xz}\\
0 & 1+e_{yy} & e_{yz}\\
0 & 0 & 1+e_{zz}
\end{array}\right]
$$
这样，对于应变的6个独立分量，每一个应变设定都有对应的变形矩阵设定，将变形矩阵与旧晶格相乘就得到了新的晶格。原子分数坐标不变，晶格改变，得到新的结构文件。如果结构文件使用的是笛卡尔坐标，则每个原子的$xyz$坐标向量都需要用变形矩阵乘得到新坐标。

默认设置下，每一个应变独立分量进行4个取值：[-0.01, -0.005, 0.005, 0.01]，对应4个变形矩阵；总共6个独立分量，共$4\times6=24$个变形矩阵。将这24个变形矩阵逐个作用于结构优化后的结构文件，得到24个变胞结构文件。

### 进行固定晶格优化
对每一个生成的变胞结构文件，进行固定晶格的结构优化，获得总能量和应力。  
如果使用ABACUS，将INPUT文件中的calculation设置改为relax、其它不变即可。  
如果使用LAMMPS，将in.lammps文件中所有box/relax设置删掉，保留一行minimize命令即可。

固定晶格的ABACUS输入文件INPUT设置示例如下
```
INPUT_PARAMETERS
#Parameters (General)
suffix         ABACUS
pseudo_dir     /home/lijh/abacus/selected-Dojo/Pseudopotential/
orbital_dir    /home/lijh/abacus/selected-Dojo/selected_Orbs/
nspin          1    # 1 or 2 or 4. The number of spin components of wave functions. 1 for Spin degeneracy, 2 for Collinear spin polarized, 4 for SOC. Default 1
symmetry       1    # 0 or 1. 1 for open, default 1.
esolver_type   ksdft    # ksdft, ofdft, sdft, tddft, etc. Choose the energy solver. Default ksdft.
ks_solver      genelpa    # Choose the diagonalization method for the Hamiltonian matrix expanded in a certain basis set. Default cg for pw. Default genelpa for lcao.

#Parameters (Methods)
ecutwfc        100
scf_thr        1e-8
scf_nmax       300
relax_nmax     200
relax_method   cg    # Conjugate gradient algorithm.
force_thr_ev   0.01
stress_thr     0.5
basis_type     lcao
calculation    relax    # structure relaxation calculations
dft_functional PBE

#Parameters (Smearing)
smearing_method gaussian    # recommend to set fixed for non-conductors, gaussian for semi-conductors, mp for metal. Default gaussian
smearing_sigma  0.010    # Rydberg, energy range for smearing, default 0.015.

#Parameters (charge mixing and k points spacing)
kspacing       0.1
mixing_beta     0.5    # The formula for charge mixing be written as rho_new = rho_old + beta * rho_update. Default 0.8 for nspin=1, 0.4 for nspin=2 and nspin=4

#Parameters (File)
cal_force      1
cal_stress     1
```
固定晶格的LAMMPS输入文件in.lammps设置示例如下
```
clear
units    metal
dimension    3
boundary    p p p
box    tilt large

read_data    conf.lmp

neigh_modify    every 1 delay 0 check no

pair_style    deepmd    graph.pb
pair_coeff * *

compute    mype all pe
thermo    100
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz lx ly lz vol c_mype
dump    1 all custom 100 dump.relax id type xs ys zs fx fy fz
write_data    CONTCAR.lmp

min_style    cg
minimize    0 1e-10 5000 500000

variable    E equal "c_mype"
variable    Pxx equal pxx
variable    Pyy equal pyy
variable    Pzz equal pzz
variable    Pyz equal pyz
variable    Pxz equal pxz
variable    Pxy equal pxy

print "All done"
print "Final energy = ${E}"
print "Final Stress (xx yy zz yz xz xy) = ${Pxx} ${Pyy} ${Pzz} ${Pyz} ${Pxz} ${Pxy}"
```
输入文件准备好后，对24个结构文件提交执行计算。

### 拟合得到弹性常数矩阵结果
24个应变，对应24个结构，计算完获得24个应力stress数据。24对数据每4对为一组对应1个独立分量，在voigt notation下拟合矩阵方程
$$
\left(\begin{array}{c}
\sigma_{1}\\
\sigma_{2}\\
\sigma_{3}\\
\sigma_{4}\\
\sigma_{5}\\
\sigma_{6}
\end{array}\right)=\left[\begin{array}{cccccc}
C_{11} & C_{12} & C_{13} & C_{14} & C_{15} & C_{16}\\
C_{21} & C_{22} & C_{23} & C_{24} & C_{25} & C_{26}\\
C_{31} & C_{32} & C_{33} & C_{34} & C_{35} & C_{36}\\
C_{41} & C_{42} & C_{43} & C_{44} & C_{45} & C_{46}\\
C_{51} & C_{52} & C_{53} & C_{54} & C_{55} & C_{56}\\
C_{61} & C_{62} & C_{63} & C_{64} & C_{65} & C_{66}
\end{array}\right]\left(\begin{array}{c}
\epsilon_{1}\\
\epsilon_{2}\\
\epsilon_{3}\\
\epsilon_{4}\\
\epsilon_{5}\\
\epsilon_{6}
\end{array}\right)
$$
按照voigt notation，$1\rightarrow xx,\quad2\rightarrow yy,\quad3\rightarrow zz,\quad4\rightarrow yz,\quad5\rightarrow xz,\quad6\rightarrow xy$.

本程序中，线性拟合使用`numpy.linalg.lstsq()`函数，逐行拟合$C$矩阵。

## 能量差分计算应力Stress
ABACUS软件和LAMMPS软件都有计算Stress输出的功能，但往往精度有限。本程序同时提供了能量有限差分法计算得到应力Stress的功能。

根据应力张量的定义，它是能量对单位体积应变张量的偏导数，分量写为
$$
𝜎_{𝛼𝛽}=-\frac{1}{\Omega}\left(\frac{\partial E}{\partial\varepsilon_{𝛼𝛽}}\right)_{\varepsilon=0}
$$
其中$\Omega$是初始体积，$E$为总能（严格来说应该是弹性能，但是因为变形导致的总能变化即是弹性能变化，可以替代）。求导变差分得到
$$
𝜎_{𝛼𝛽}=-\frac{1}{\Omega}\frac{E_{𝛼𝛽}-E_{0}}{\delta\varepsilon_{𝛼𝛽}}
$$
其中$E_{𝛼𝛽}$是进行了微小变胞后结构的总能，$E_{0}$是初始结构的总能，$\delta\varepsilon_{𝛼𝛽}$为1个微小变胞步长的应变值，本程序中此值默认值为0.0001。

上式展示的是单方向差分计算，为提高精度，本程序使用中心差分，公式为
$$
𝜎_{𝛼𝛽}=-\frac{1}{\Omega}\frac{E_{𝛼𝛽}-E_{-𝛼𝛽}}{2\delta\varepsilon_{𝛼𝛽}}
$$

能量有限差分计算应力stress的流程为：
1. 对relax完的结构进行微小变胞操作，6个独立分量，正负各一次，生成12个微小变胞结构。
2. 对微小变胞后结构进行单步计算。如果使用ABACUS，进行一轮自洽scf计算；如果使用LAMMPS，使用`run    0`进行单步计算。
3. 计算得到微小变胞结构能量，即可以代入差分公式计算得到应力，总共6个分量的值。

值得一提的是，对于DFT计算来说，选择使用能量差分计算应力stress，相当于每个变胞结构要多进行12次scf计算，总共多出$24\times12=288$次自洽计算，这是相当大的额外计算量。

## 本程序使用指南

### python环境需求
只需要`numpy`和`monty`。

本程序开发所用的版本为`numpy 2.2.4`，`monty 2024.7.30`，`python 3.12.0`。仓库的Venv文件夹即为本程序开发时使用的虚拟环境。

在另一台设备上的`numpy 1.26.4`，`monty 2024.2.26`，`python 3.10.13`环境做了迁移性测试，无bug无error。

坦诚说，本程序工作约等于重复造轮子，把pymatgen程序包已经实现的一个功能用Kittel的规范复现，并复现加入abacus-test中已经成熟的能量有限差分应力stress算法。因此并没有调用什么其它现成的科学计算程序包。

### 配置文件编写格式
本程序需要编写一个python dict字典格式的配置文件，用来设置ABACUS计算或LAMMPS计算、和生成变胞的参数。文件需要是.json格式，在程序中被`monty.serialization.loadfn()`函数读取。

#### 使用ABACUS的配置文件样例如下，不妨命名为`abacus_config.json`
```json
{
  "calculator": "abacus",
  "OMP_NUM_THREADS": 1,
  "MPIRUN_NUM_PROC": 2,
  "parameters": {
    "pseudo_dir": "/selected-Dojo/Pseudopotential/",
    "orbital_dir": "/selected-Dojo/selected_Orbs/",
    "dft_functional": "PBE",
    "nspin": 1,
    "symmetry": 1,
    "basis_type": "lcao",
    "ks_solver": "genelpa",
    "ecutwfc": 100,
    "scf_thr": 1e-8,
    "scf_nmax": 300,
    "relax_method": "cg",
    "relax_nmax": 200,
    "force_thr_ev": 0.01,
    "stress_thr": 0.5,
    "chg_extrap": "first-order",
    "kspacing": 0.1,
    "mixing_type": "broyden",
    "mixing_beta": 0.5,
    "smearing_method": "gaussian",
    "smearing_sigma": 0.010
  },
  "stru_files": [
    "/Elastic/test_diamond/cell-relax-work/cell-relax/cell-relaxed.STRU"
  ],
  "relax_log_dir": [
    "/Elastic/test_diamond/cell-relax-work/cell-relax/OUT.ABACUS/running_cell-relax.log"
  ],
  "norm_deform": 0.01,
  "shear_deform": 0.01,
  "run_efd": 0,
  "small_deform": 0.0001
}
```
此配置文件示例包括了几乎所有本程序目前支持的设置项。配置文件支持的所有参数设置说明如下

|设置项|说明|允许设置值|默认值|
|------|----|------------|------|
|"calculator"|是用LAMMPS还是ABACUS做计算，决定了输入输出文件生成的格式|"abacus", "lammps"|无。必须显式设置|
|"OMP_NUM_THREADS"|使用openmpi并行的线程数| - |1|
|"MPIRUN_NUM_PROC"|使用mpirun并行的进程数| - |1|
|"parameters"|一个字典，内容为主要的ABACUS输入INPUT文件设置| - | - |
|"stru_files"|一个list，内容为（一个或多个）进行完结构优化的文件路径| - |必须显式设置|
|"relax_log_dir"|一个list，内容为"stru_files"中对应文件结构优化任务的输出log路径，要求一一对应| - |必须显式设置|
|"norm_deform"|进行正向应变的大小| - |0.01|
|"shear_deform"|进行剪切应变的大小| - |0.01|
|"run_efd"|设置是否进行能量差分计算应力Stress。0为不进行，1为进行|0, 1|0|
|"small_deform"|能量差分计算应力stress时，进行微小应变的大小| - |0.0001|

|"parameters"内容项|说明|允许设置值|默认值|
|------------------|----|------------|------|
|"pseudo_dir"|赝势文件夹路径，需要与提供的结构文件中的设置配合| - |""|
|"orbital_dir"|轨道文件夹路径，需要与提供的结构文件中的设置配合| - |""|
|"dft_functional"|计算中使用的泛函方法| - |取决于提供的赝势文件|
|"nspin"|波函数的自旋分量数。1为自旋简并，2为共线的自旋极化，4为非共线的自旋极化（自旋轨道耦合）计算| 1, 2, 4 |1|
|"symmetry"|ABACUS计算中是否开启对称分析。1为进行对称性分析，0为只考虑时间反演对称性，-1为不考虑任何对称性。| -1, 0, 1 |1|
|"basis_type"|ABACUS计算时用到的基组，pw为平面波基组，lcao为数值原子轨道基组| "pw", "lcao" |"lcao"|
|"ks_solver"|在使用的基组下进行展开哈密顿矩阵的方法| "cg", "bpcg", "dav", "dav_subspace", "lapack", "genelpa", "scalapack_gvx", "cusolver", "cusolvermp", "elpa" |取决于基组和计算环境配置|
|"pw_diag_ndim"|进行电子自洽迭代的Davidson算法的工作空间维度（波函数波包的数量，至少需要2）| - |4|
|"pw_diag_nmax"|使用平面波基组时，"cg"、"bpcg"、"dav"、"dav_subspace"方法的最大迭代步数| - |40|
|"ecutwfc"|波函数能量截断，单位Rydberg| - |100|
|"scf_thr"|电子自洽迭代的收敛阈值| - |1e-7|
|"scf_nmax"|电子自洽迭代的最大步数| - |100|
|"relax_method"|进行固定晶格结构优化步骤中所用到的优化算法|"cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"|"cg"|
|"relax_nmax"|最大离子迭代步数| - |100|
|"force_thr_ev"|结构优化中的力收敛判据，单位为eV| - |0.01|
|"stress_thr"|结构优化中的应力stress收敛判据，单位为KBar| - |0.5|
|"chg_extrap"|结构优化计算时进行密度外推的方法。atomic为原子外推，first-order为一阶外推，second-order为二阶外推|"atomic", "first-order", "second-order"|"first-order"|
|"kspacing"|设置K空间取点的最小间距。用于生成KPT文件| - |0.1|
|"mixing_type"|自洽计算中使用的电荷混合方法|"plain", "pulay", "broyden"|"broyden"|
|"mixing_beta"|电荷混合更新系数| - |0.8 if nspin=1, 0.4 if nspin=2 or nspin=4|
|"smearing_method"|DFT计算中用到的轨道占据与费米能展宽方法|"fixed", "gauss" or "gaussian", "mp", "mp2", "mv" or "cold", “fd"|"gaussian"|
|"smearing_sigma"|能量展宽范围，单位Rydberg| - |0.01|

注意：
1. 请提供赝势和轨道文件的**绝对路径**，可以直接在结构文件里写好，也可以在配置文件里设定文件夹配合结构文件里设定文件名。本程序与vaspkit等软件生成文件的方式不同，不会对给出的赝势和轨道文件进行复制或移动操作，也不会改写相对路径，只会复制或移动或生成结构文件。因此需要一开始确定好绝对路径，防止abacus任务提交时报错。
2. 请确保在配置文件中"stru_files"提供的list里，每一个结构文件的**文件名不同**。可以是相同目录下的不同文件名，也可以是不同目录下的不同文件名，但不能是不同目录下的相同文件名。此限制由本程序自身缺陷导致，因为本程序生成工作目录是根据结构文件名命名，如果提供的结构文件名相同会导致不同结构的工作目录相互覆盖。日后更新计划改正这一问题。  
3. 本文档中给出的部分默认参数设置与ABACUS官方文档不同。当使用本程序生成输入文件且配置文件中有缺省设置时，默认计算参数请以本文档为准。

#### 使用LAMMPS + DP势函数模型的配置文件样例如下，不妨命名为`lammps_config.json`
```json
{
  "calculator": "lammps",
  "MPIRUN_NUM_PROC": 2,
  "interaction": {
    "method": "deepmd",
    "model": "graph.pb",
    "type_map": {
      "Hf": 0,
      "Zr": 1,
      "Y": 2,
      "Al": 3,
      "O": 4
    }
  },
  "parameters": {
    "etol": 0,
    "ftol": 1e-10,
    "maxiter": 5000,
    "maximal": 500000
  },
  "stru_files": [ "confs/t_P42nmc/CONTCAR.lmp" ],
  "relax_log_dir": [ "confs/t_P42nmc/relax_task/log.lammps" ],
  "norm_deform": 0.01,
  "shear_deform": 0.01,
  "run_efd": 0,
  "small_deform": 0.0001
}
```
此配置文件示例包括了所有本程序目前支持的设置项。说明如下
|设置项|说明|允许设置值|默认值|
|------|----|----------|------|
|"calculator"|是用LAMMPS还是ABACUS做计算，决定了输入输出文件生成的格式|"abacus", "lammps"|无。必须显式设置|
|"MPIRUN_NUM_PROC"|使用mpirun并行的进程数| - |1|
|"interaction"|一个字典，设置LAMMPS在MD模拟中使用的相互作用势信息。目前仅支持deepmd|"deepmd"|无。必须显式设置|
|"parameters"|一个字典，内容为LAMMPS进行结构优化的参数| - |{"etol": 0, "ftol": 1e-10, "maxiter": 5000, "maximal": 500000}|
|"stru_files"|一个list，内容为（一个或多个）进行完结构优化的文件路径| - |必须显式设置|
|"relax_log_dir"|一个list，内容为"stru_files"中对应文件结构优化任务的输出log路径，要求一一对应| - |必须显式设置|
|"norm_deform"|进行正向应变的大小| - |0.01|
|"shear_deform"|进行剪切应变的大小| - |0.01|
|"run_efd"|设置是否进行能量差分计算应力Stress。0为不进行，1为进行|0, 1|0|
|"small_deform"|能量差分计算应力stress时，进行微小应变的大小| - |0.0001|

|"interaction"内容项|说明|允许设置值|默认值|
|-------------------|----|------------|------|
|"method"|使用的势函数名。目前仅支持deepmd|"deepmd"|无。必须显式设置|
|"model"|dp势函数模型文件路径。可提供相对路径或绝对路径| - |无。必须显式设置|
|"type_map"|与训练dp势函数时一致，对于deepmd计算必须提供| - |无。必须显式设置|

|"parameters"内容项|说明|允许设置值|默认值|
|------------------|----|------------|------|
|"etol"|结构优化的能量收敛阈值，无量纲。设为0表示不使用此判据| - |0|
|"ftol"|结构优化的力收敛阈值，单位eV/Angstrom。设为0表示不使用此判据| - |1e-10|
|"maxiter"|结构优化中更新原子位置的最大迭代步数| - |5000|
|"maximal"|结构优化计算中调用势函数的最大次数，用于控制计算成本| - |500000|

注意：
1. 与LAMMPS结合，本程序目前只支持deepmd势函数模型。后续有添加LJ势和EAM势的计划。
2. "interaction"["type_map"]这一字典设置一定要与训练dp模型时相同。
3. 与前文ABACUS部分提到的注意事项相同，请确保在配置文件中"stru_files"提供的list里，每一个结构文件的**文件名不同**。

一般来说，对于一组计算，准备1个配置文件就足够。

### 程序使用步骤
在准备好配置文件后，命令行输入
```
python3 main.py config.json
```
即开始执行整个计算流程，等待计算完成即可。  
实际使用中，请把`main.py`和`config.json`分别替换成脚本文件和配置文件的正确路径。

以下部分介绍逐步执行的步骤。

#### 变胞结构生成
在准备好配置文件后，命令行输入
```
python3 make.py config.json
```
即开始创建工作目录`work/`，并生成每个待计算结构的变胞结构。  
实际使用中，请把`make.py`和`config.json`分别替换成脚本文件和配置文件的正确路径。

#### 提交固定晶胞结构优化的计算任务
通过bash脚本单独另外完成。下面提供一个示例，不妨命名为`run_relax_tasks.sh`
```bash
cd work/

for stru in ./*/; do
    cd $stru
	for task in ./*/; do
	    cd $task
		# choose one command that fits the calculator
		# OMP_NUM_THREADS=1 mpirun -n 8 abacus
		# mpirun -np 8 lmp -i in.lammps
		cd ../
	done
	cd ../
done
```

#### （可选）进行能量差分计算应力stress
在固定晶胞结构优化relax全部做完后，如果如果要进行能量差分计算应力stress，首先要生成微小变胞结构文件。命令行输入
```
python3 EFD_make.py config.json
```
即会在每个`task.*/`任务目录下生成一个`EFD/`目录，在里面准备好6组12个微小变胞的单步计算任务文件。在结构文件任务目录下还会生成一个`EFD_task/`目录，准备好初始结构的12个微小变胞计算任务文件。此外，在结构文件任务目录下会生成一个空的`EFD/`目录，提示此任务进行了能量差分计算应力stress。  
实际使用中，请把命令里`EFD_make.py`和`config.json`分别替换成脚本文件和配置文件的正确路径。

接下来，提交计算任务。用一个bash脚本完成，不妨命名为`run_efd_task.sh`
```bash
cd work/

for stru in ./*/; do
    cd $stru
    cd EFD_task/
	for i in ./*/; do
		cd $i
		# choose one command that fits the calculator
		# OMP_NUM_THREADS=1 mpirun -n 8 abacus
		# mpirun -np 8 lmp -i in.lammps
		cd ../
	done
    cd ../

	for task in task.*/; do
		cd $task
		cd EFD
		for j in ./*/; do
			cd $j
			# choose one command that fits the calculator
		    # OMP_NUM_THREADS=1 mpirun -n 8 abacus
		    # mpirun -np 8 lmp -i in.lammps
			cd ../
		done
		cd ../../
	done
	cd ../
done
```

所有单步计算完成后，命令行输入
```
python3 EFD_post.py config.json
```
即会根据能量结果和中心差分公式计算每个变胞结构和初始结构的应力stress，以numpy矩阵形式存储到一个`stress.json`文件里。  
实际使用中，请把命令里`EFD_post.py`和`config.json`分别替换成脚本文件和配置文件的正确路径。

#### 拟合弹性常数矩阵，计算各模量
前述操作完成，计算任务全部完成后，命令行输入
```
python3 post.py config.json
```
即会根据计算结果拟合弹性常数矩阵，以及体积模量、杨氏模量、剪切模量、泊松比。  
实际使用中，请把命令里`post.py`和`config.json`分别替换成脚本文件和配置文件的正确路径。

计算结果存储在结构文件任务目录里的`elastic_constant`目录下，有numpy矩阵格式的`elastic_constant.json`文件和txt文本格式的`result.txt`文件。以ABACUS计算金刚石为例，`result.txt`文件内容如下
```
Elastic constant (GPa): 
==============================

   1051.73    115.50    115.50      0.00      0.00      0.00
    115.50   1051.73    115.50      0.00      0.00      0.00
    115.50    115.50   1051.73      0.00      0.00      0.00
      0.00      0.00      0.00    561.17      0.00      0.00
      0.00      0.00      0.00      0.00    561.17      0.00
      0.00      0.00      0.00      0.00      0.00    561.17

==============================

Bulk Modulus Bv = 427.58 (GPa)
Youngs Modulus Ev = 1116.00 (GPa)
Shear Modulus Gv = 523.95 (GPa)
Poisson Ratio = 0.06

```

# 结尾
这是一个完整的程序，经过多个计算任务测试，代码上没有恶性bug或error，功能可用。

数据准确性上，进行了与vaspkit + vasp，vaspkit + abacus，pymatgen + abacus对比的工作，相同输入参数下结果匹配。

代码和公式的正确性会持续验证，持续更新。