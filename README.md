# 伊辛模型

**[English](README_en.md) | [中文](README.md)**

基于伊辛模型的 SRAM-CIM 退火芯片算法。

> [!NOTE]
> 这是一个 **MultiSpin** 实现（多自旋态的 Potts 模型）。对于经典伊辛模型的实现，请参考分支 [**BinarySpin**](https://github.com/yiyaowen/IsingModel/tree/BinarySpin)。

## 环境配置

### 版本

本项目在 Windows 11 上使用 Python 3.10.11 64-bit 进行开发与测试，理论上 3.10.* 及后续版本均可正常运行。使用其它版本的 Python 运行脚本可能会遇到依赖包（例如 matplotlib）版本不兼容的情况。此外脚本中可选的导出 MP4 文件使用了 FFMpeg 库进行解码，因此你可能需要下载对应平台的 FFMpeg 动态库并设置好相关的环境变量（具体的操作步骤，此处省略 100 字 (¬‿¬)）。

### 依赖

```bat
pip install requirements.txt
```

## 快速上手

伊辛模型可以用来模拟晶体中自旋态变化导致的相变现象，这也是其最初的应用场景。随着人工智能的发展，该模型也被交叉应用到诸如图像处理、神经网络和量子算法等各个领域，例如：1）通过与马尔可夫随机场与全变分（Total Variation）算法的结合，伊辛模型可以用来降低图像的噪声；2）伊辛模型的求解（即最低能量态 Ground State 的获取）已经被证明是一个 NP-Complete 问题，因此可以将其它 NP 问题（例如最大割集 Max-Cut 问题）规约到该模型上，然后再通过量子计算等进行求解。

与量子计算机不同，通用计算机上的 CPU/GPU 以及专用芯片上的 CMOS 电路本质上还是通过模拟退火的方式来求解伊辛模型。模拟退火的过程可以使用蒙特卡洛采样来实现，例如常用的 Metropolis 和 Gibbs 等采样方法。本项目使用 CPU + Metropolis 来求解伊辛模型，相关的算法也可以进一步映射到专用的 CMOS 退火芯片上并使用 SRAM-CIM 等技术来加速求解。

> [!IMPORTANT]
> 这里不再详细介绍伊辛模型的具体原理，因此在继续阅读之前，请先确保你对伊辛模型有一些基本的了解。

### Hello World

默认情况下，一个网格中的所有 J-Coupling 系数都被初始化为 +1，因此最后的结果往往表现为系统中出现不同形状的随机色块（这是合理的，因为 J=+1 使相邻自旋态趋同，而 J=-1 使相邻自旋态互斥）。

```bat
python IsingModel.py
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/hello_world.gif" width=50%/>

默认为 Potts 模型设置了 256 种自旋状态，你也可以通过设置 `-smin` 和 `-smax` 来指定自旋态的离散空间。例如通过如下命令来求解经典伊辛模型的随机初态系统，从而生成二值图。

```bat
python IsingModel.py -tmin 0 -tmax 1
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/binary_spin.gif" width=50%/>

### Max-Cut 问题

Max-Cut 问题的求解过程可以等价为对顶点进行染色分组，而这可以直接映射到伊辛模型的不同自旋态上。一般来说，将实际的 Max-Cut 问题规约到伊辛模型上进行求解得到的结果是很难可视化的，为了演示伊辛模型优雅的求解过程，可以通过人为设计特殊的 Max-Cut 问题，使其最小能量态正好对应到某些具有实际意义的图像（例如一组文字）。

#### 生成数字 1

```bat
python IsingModel.py -w 28 -h 28 -tmax 20 -tmin 1 -smin 0 -smax 1 -jj j-coupling/number_1.txt
```

<table><tr>
<td>J-Couplings</td><td>求解过程</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/j-coupling/number_1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/gen_num_1.gif"/></td>
</tr></table>

#### 生成单词 SEMI

```bat
python IsingModel.py -w 200 -h 100 -tmax 20 -tmin 1 -smin 0 -smax 1 -jj j-coupling/word_SEMI.txt -e 50_000_000
```

随着图像尺寸的增大，待求解问题的状态空间也不断增大（例如 200x100 的二值图像具有 200x100x2=4w 种不同的状态），此时需要相应地增大采样次数（例如设置 -e 50_000_000）来提高收敛的概率。值得注意的是每次仿真并不保证能求解得到正确的答案，这往往取决于退火算法的设计与迭代次数的设置，目标是避免落入局部最小值，尽量减小系统的最终能量。

<table><tr>
<td>J-Couplings</td><td>OK</td><td>Fail</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/j-coupling/word_SEMI.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/ok_word_SEMI.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/fail_word_SEMI.gif"/></td>
</tr></table>

### 二值图降噪

我们可以将全变分（TV）算法映射到伊辛模型，从而实现对图像的降噪。值得注意的是 TV 算法的目标是一个凸优化问题（即存在唯一的全局最小值），我们可以通过梯度下降等算法来很好地求解该问题，那我们为何还要尝试需要模拟退火的伊辛模型呢？这是因为诸如梯度下降等算法需要计算偏导数，从而严重依赖于强大的浮点算力，而伊辛模型的采样过程十分简单，这使得它在电路实现上具有优势，因此相关的研究是有意义的。

#### 测试单词 ABC

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -smin 0 -smax 1 -js c -jj tv-denoise/ABC/j_noise_ABC.txt -hl 4 -hh tv-denoise/ABC/h_noise_ABC.txt
```

<table><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/ABC/ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/ABC/noise_ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_word_ABC.gif"/></td>
</tr></table>

#### 测试数字集

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -smin 0 -smax 1 -js c -jj tv-denoise/numbers/system/j_noise_0.txt -hl 4 -hh tv-denoise/numbers/system/h_noise_0.txt
```

可以更改 `j_noise_0` 和 `h_noise_0` 来测试不同的数字。

<table><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/0.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_0.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_0.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_1.gif"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/2.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_2.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_2.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/3.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_3.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_3.gif"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/4.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_4.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_4.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/5.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_5.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_5.gif"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/6.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_6.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_6.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/7.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_7.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_7.gif"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/8.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_8.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_8.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/rawpng/9.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/numbers/noise/noise_9.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_9.gif"/></td>
</tr></table>

该过程的能量变化如下：

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_energy.gif" width=50%/>

#### 测试 maze128

<table><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/maze128/maze128.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/maze128/noise_maze128.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_maze128.gif"/></td>
</tr></table>

### 灰度图降噪

#### 泊松噪声

这是各种图像传感器输出的原始图片中常见的一种噪声，并且很难通过简单的手段去除。

```bat
python IsingModel.py -w 256 -h 256 -a plateau -tmax 20 -tmin 1 -smin 0 -smax 255 -js c -jj tv-denoise\lenna\poisson\j_poisson_lenna.txt -hl 4 -hh tv-denoise\lenna\poisson\h_poisson_lenna.txt -e 10_000_000
```

<table><tr>
<th>退火过程</th>
<th>能量变化</th>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_anim_poisson_lenna.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_energy_poisson_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>原图</th>
<th>泊松噪声</th>
<th><font color="green">伊辛模型</font></th>
<th>双边滤波</th>
<th>高斯滤波</th>
</tr><tr>
<th>PSNR</th>
<td>NA</td>
<td>26.96</td>
<td><font color="green">29.86</font></td>
<td>28.34</td>
<td>24.42</td>
</tr><tr>
<th>SSIM</th>
<td>1.0</td>
<td>0.6587</td>
<td><font color="green">0.8369</font></td>
<td>0.8343</td>
<td>0.7306</td>
</tr><tr>
<th>结果</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/poisson_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_spins_poisson_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/bilateral_poisson_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/gaussian_poisson_lenna.png"/></td>
</tr></table>

注：双边滤波参数 d=2, sigmaColor=100；高斯滤波参数 size=2

事实上，当双边滤波器取 sigmaColor=75 和更小的值时，可以得到 PSNR 和 SIMM 小优于伊辛模型的结果；这里我们想说明的是伊辛模型可以得到和双边滤波类似的结果（即去除噪声的同时仍然保留边缘细节），但双边滤波依赖于更强大的浮点算力（众所周知 bilateral 慢的很，(¬‿¬)）

#### 椒盐噪声

```bat
python IsingModel.py -w 256 -h 256 -a plateau -tmax 20 -tmin 1 -smin 0 -smax 255 -js c -jj tv-denoise\lenna\salt_pepper\j_salt_pepper_lenna.txt -hl 4 -hh tv-denoise\lenna\salt_pepper\h_salt_pepper_lenna.txt -e 10_000_000
```

<table><tr>
<th>退火过程</th>
<th>能量变化</th>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_anim_salt_pepper_lenna.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_energy_salt_pepper_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>原图</th>
<th>椒盐噪声</th>
<th><font color="green">伊辛模型</font></th>
<th>中值滤波-3</th>
<th>中值滤波-5</th>
</tr><tr>
<th>PSNR</th>
<td>NA</td>
<td>14.04</td>
<td><font color="green">29.64</font></td>
<td>28.85</td>
<td>26.48</td>
</tr><tr>
<th>SSIM</th>
<td>1.0</td>
<td>0.2012</td>
<td><font color="green">0.9257</font></td>
<td>0.9090</td>
<td>0.8276</td>
</tr><tr>
<th>结果</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/salt_pepper_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_spins_salt_pepper_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/median3_salt_pepper_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/median5_salt_pepper_lenna.png"/></td>
</tr></table>

## 参数列表

```bat
python IsingModel.py [arguments]
```

| Long             | Short | Value                |
|------------------|-------|----------------------|
| --width          | -w    | 28                   |
| --height         | -h    | 28                   |
| --graph-type     | -g    | [nearest, king]      |
| --annealing      | -a    | [lin, exp, cos] etc. |
| --anneal-temp1   | -tmax | 20                   |
| --anneal-temp2   | -tmin | 10                   |
| --init-state     | -js   | [r, u-1, u+1, c]     |
| --spin-space0    | -smin | 0                    |
| --spin-space1    | -smax | 255                  |
| --j-coupling     | -jj   | path/to/j_system.txt |
| --h-lambda       | -hl   | 1                    |
| --h-coupling     | -hh   | path/to/h_system.txt |
| --sample-count   | -c    | 1                    |
| --epochs         | -e    | 1_000_000            |
| --output-video   | -o    | output-files/a.gif   |
| --frame-scale    | -s    | 1                    |
| --frame-rate     | -f    | 5                    |
| --anim-duration  | -d    | 3                    |
| --parallel-count | -p    | 1                    |

## 辅助工具

### annealing/DrawCurves.py

绘制不同的退火温度曲线。

```bat
cd annealing
python DrawCurves.py
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/annealing/curves.png" width=50%/>

### Convert.py

将目标问题（图像或者其它文件）规约到伊辛模型（文本文件）。例如输入 `binary.png`，输出 `j_binary.txt` 和 `h_binary.txt`（可选），得到的初始自旋态和 J-Coupling 系数（包含在 `j_binary.txt` 中）和 H-Coupling 系数（包含在 `h_binary.txt` 中）可以进一步作为 `IsingModel.py` 的输入。使用文本文件存储系统参数可以方便手动修改和调试，特别是针对不同的初态和耦合系数可以快速地进行测试，代价是较大的文件体积（对于现代计算机来说不算什么）。

```bat
python Convert.py -p Generate binary.png
python Convert.py -p Denoise -c monochrome noise_binary.png
python Convert.py -p Denoise -c grayscale noise_lenna.png
```

| Long         | Short | Value                   |
|--------------|-------|-------------------------|
| --problem    | -p    | [Generate, Denoise]     |
| --graph-type | -g    | [nearest, king]         |
| --color-type | -c    | [monochrome, grayscale] |

### AddNoise.py

为灰度图像添加噪声。下面的命令会生成 `noise_raw.png`：

```bat
python AddNoise.py [-t salt_pepper] [-p 0.1] raw.png
```

| Long          | Short | Value                  |
|---------------|-------|------------------------|
| source        |       | path/to/raw_image      |
| --noise-type  | -t    | [salt_pepper, poisson] |
| --noise-param | -p    | 0.1                    |

### AddFilter.py

对灰度图像进行滤波。下面的命令会应用一个 3x3 的中值滤波：

```bat
python AddFilter.py [-t median] [-p 1] noise.png
```

| Long           | Short | Value                         |
|----------------|-------|-------------------------------|
| source         |       | path/to/noise_image           |
| --filter-type  | -t    | [median, gaussian, bilateral] |
| --filter-param | -p    | 1                             |

### CalcMetrics.py

计算不同噪声图像相对于基准图像的 PSNR 和 SSIM 参数值：

```bat
python CalcPSNR.py <base_image> <noisy_image1> [<noisy_image2> ...]
```
