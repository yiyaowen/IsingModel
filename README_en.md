# Ising Model

**[English](README_en.md) | [中文](README.md)**

SRAM-CIM annealing chip algorithm based on the Ising model.

> [!NOTE]
> This is a **MultiSpin** implementation (Potts model with multiple spin states). For the classical Ising model implementation, please refer to the [**BinarySpin**](https://github.com/yiyaowen/IsingModel/tree/BinarySpin) branch.

## Environment Setup

### Versions

This project was developed and tested on Windows 10/11 using Python 3. It should run normally on Python 3.10.* and later versions. Using other Python versions may result in syntax and dependency package (e.g., matplotlib) compatibility issues. The specific development and testing platform information is as follows:

* Windows 10 (22H2) & Python 3.13.3 64-bit
* Windows 11 (24H2) & Python 3.10.11 64-bit

The script uses PIL (PillowWriter) by default to export GIF files. For optional MP4 export, it uses the FFMpeg library for decoding. To export MP4 files, you need to download the FFMpeg dynamic library for your platform and set the relevant environment variables (specific steps omitted here (¬‿¬)).

### Dependencies

```bat
pip install requirements.txt
```

## Quick Start

The Ising model can simulate phase transition phenomena caused by spin state changes in crystals, which was its original application. With the development of artificial intelligence, this model has been cross-applied in fields such as image processing, neural networks, and quantum algorithms. For example: 1) Combined with Markov Random Fields and Total Variation algorithms, the Ising model can reduce image noise; 2) Solving the Ising model (obtaining the ground state) has been proven to be an NP-Complete problem, so other NP problems (e.g., Max-Cut) can be reduced to this model and then solved through quantum computing.

Unlike quantum computers, CPUs/GPUs in general-purpose computers and CMOS circuits in specialized chips essentially solve the Ising model through simulated annealing. The simulated annealing process can be implemented using Monte Carlo sampling methods such as Metropolis and Gibbs. This project uses CPU + Metropolis to solve the Ising model, and related algorithms can be further mapped to dedicated CMOS annealing chips and accelerated using technologies like SRAM-CIM.

> [!IMPORTANT]
> The specific principles of the Ising model will not be detailed here. Before continuing, please ensure you have a basic understanding of the Ising model.

### Hello World

By default, all J-Coupling coefficients in a grid are initialized to +1, so the final result often appears as random color blocks of different shapes in the system (this is reasonable since J=+1 causes adjacent spin states to align, while J=-1 causes mutual exclusion).

```bat
python IsingModel.py
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/hello_world.gif" width=50%/>

The Potts model defaults to 256 spin states. You can also specify the discrete space of spin states by setting `-smin` and `-smax`. For example, use the following command to solve a classical Ising model system with random initial states, generating a binary image:

```bat
python IsingModel.py -tmin 0 -tmax 1
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/binary_spin.gif" width=50%/>

### Max-Cut Problem

The solution process of the Max-Cut problem can be equivalent to coloring and grouping vertices, which can be directly mapped to different spin states in the Ising model. Generally, visualizing the results of reducing actual Max-Cut problems to the Ising model is difficult. To demonstrate the elegant solving process of the Ising model, we can design special Max-Cut problems whose ground states correspond to meaningful images (e.g., text).

#### Generating Digit 1

```bat
python IsingModel.py -w 28 -h 28 -tmax 20 -tmin 1 -smin 0 -smax 1 -jj j-coupling/number_1.txt
```

<table><tr>
<td>J-Couplings</td><td>Solving Process</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/j-coupling/number_1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/gen_num_1.gif"/></td>
</tr></table>

#### Generating Word SEMI

```bat
python IsingModel.py -w 200 -h 100 -tmax 20 -tmin 1 -smin 0 -smax 1 -jj j-coupling/word_SEMI.txt -e 50_000_000
```

<table><tr>
<td>J-Couplings</td><td>OK</td><td>Fail</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/j-coupling/word_SEMI.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/ok_word_SEMI.gif"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/fail_word_SEMI.gif"/></td>
</tr></table>

As image size increases, the state space of the problem to be solved also expands (e.g., a 200x100 binary image has 200x100x2=40k different states). Accordingly, the sampling count needs to be increased (e.g., set `-e 50_000_000`) to improve convergence probability. Note that each simulation does not guarantee a correct solution, which depends on the annealing algorithm design and iteration settings. The goal is to avoid local minima and minimize the system's final energy.

### Binary Image Denoising

We can map the Total Variation (TV) algorithm to the Ising model for image denoising. Note that the TV algorithm targets a convex optimization problem (i.e., it has a unique global minimum), which can be well solved by gradient descent. So why use the Ising model requiring simulated annealing? Because algorithms like gradient descent require derivative calculations and heavily rely on strong floating-point computing power, while the sampling process of the Ising model is simple, giving it advantages in circuit implementation—making related research meaningful.

#### Testing Word ABC

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -smin 0 -smax 1 -js c -jj tv-denoise/ABC/j_noise_ABC.txt -hl 4 -hh tv-denoise/ABC/h_noise_ABC.txt
```

<table><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/ABC/ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/ABC/noise_ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_word_ABC.gif"/></td>
</tr></table>

Comparison with median filtering:

<table><tr>
<td>Ising</td><td>median3</td><td>median5</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_word_ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/median3_ABC.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/median5_ABC.png"/></td>
</tr></table>

> [!NOTE]
> Median-3 filtering struggles to remove all white dots, while median-5 causes over-expansion; the Ising model achieves a golden balance between them.

#### Testing Digit Set

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -smin 0 -smax 1 -js c -jj tv-denoise/numbers/system/j_noise_0.txt -hl 4 -hh tv-denoise/numbers/system/h_noise_0.txt
```

You can change `j_noise_0` and `h_noise_0` to test different digits.

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

Energy changes during this process:

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_energy.png" width=50%/>

Comparison with median filtering:

<table><tr>
<td>Ising</td><td>median3</td><td>median5</td>
<td>Ising</td><td>median3</td><td>median5</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_0.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_0.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_0.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_1.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_1.png"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_2.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_2.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_2.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_3.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_3.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_3.png"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_4.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_4.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_4.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_5.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_5.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_5.png"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_6.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_6.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_6.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_7.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_7.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_7.png"/></td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_8.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_8.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_8.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/tv_num_9.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median3_9.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/denoise_numbers/median5_9.png"/></td>
</tr></table>

> [!NOTE]
> Median-3 filtering struggles to remove all white dots, while median-5 causes over-expansion; the Ising model achieves a golden balance between them.

#### Testing maze128

<table><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/maze128/maze128.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/maze128/noise_maze128.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_maze128.gif"/></td>
</tr></table>

Comparison with median filtering:

<table><tr>
<td>Ising</td><td>median3</td><td>median5</td>
</tr><tr>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/tv_maze128.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/median3_maze.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/examples/median5_maze.png"/></td>
</tr></table>

> [!NOTE]
> Median-3 filtering struggles to remove all white dots, while median-5 causes over-expansion; the Ising model achieves a golden balance between them.

### Grayscale Image Denoising

#### Poisson Noise

This is a common noise in raw images from various image sensors and is difficult to remove by simple methods.

```bat
python IsingModel.py -w 256 -h 256 -a plateau -tmax 20 -tmin 1 -smin 0 -smax 255 -js c -jj tv-denoise\lenna\poisson\j_poisson_lenna.txt -hl 4 -hh tv-denoise\lenna\poisson\h_poisson_lenna.txt -e 10_000_000
```

<table><tr>
<th>Annealing Process</th>
<th>Energy Changes</th>
</tr><tr>
<td width=50%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_anim_poisson_lenna.gif"/></td>
<td width=50%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_energy_poisson_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>Poisson Noise</th>
<th>Gaussian Filtering</th>
</tr><tr>
<th>PSNR</th>
<td>26.96</td>
<td>24.42</td>
</tr><tr>
<th>SSIM</th>
<td>0.6587</td>
<td>0.7306</td>
</tr><tr>
<th>Result</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/poisson_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/gaussian_poisson_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>Ising Model</th>
<th>Bilateral Filtering</th>
</tr><tr>
<th>PSNR</th>
<td>29.86</td>
<td>28.34</td>
</tr><tr>
<th>SSIM</th>
<td>0.8369</td>
<td>0.8343</td>
</tr><tr>
<th>Result</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/ising_spins_poisson_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/poisson/bilateral_poisson_lenna.png"/></td>
</tr></table>

Note: Bilateral filter parameters: d=2, sigmaColor=100; Gaussian filter parameter: size=2

In fact, when the bilateral filter uses sigmaColor=75 or smaller values, it can achieve PSNR and SSIM slightly better than the Ising model. Here we want to show that the Ising model can achieve results similar to bilateral filtering (removing noise while preserving edge details), but bilateral filtering relies on stronger floating-point computing power (as we all know, bilateral is slow, (¬‿¬)).

#### Salt-and-Pepper Noise

```bat
python IsingModel.py -w 256 -h 256 -a plateau -tmax 20 -tmin 1 -smin 0 -smax 255 -js c -jj tv-denoise\lenna\salt_pepper\j_salt_pepper_lenna.txt -hl 4 -hh tv-denoise\lenna\salt_pepper\h_salt_pepper_lenna.txt -e 10_000_000
```

<table><tr>
<th>Annealing Process</th>
<th>Energy Changes</th>
</tr><tr>
<td width=50%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_anim_salt_pepper_lenna.gif"/></td>
<td width=50%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_energy_salt_pepper_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>Salt-and-Pepper Noise</th>
<th>Median-5 Filtering</th>
</tr><tr>
<th>PSNR</th>
<td>14.04</td>
<td>26.48</td>
</tr><tr>
<th>SSIM</th>
<td>0.2012</td>
<td>0.8276</td>
</tr><tr>
<th>Result</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/salt_pepper_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/median5_salt_pepper_lenna.png"/></td>
</tr></table>

<table><tr>
<th></th>
<th>Ising Model</th>
<th>Median-3 Filtering</th>
</tr><tr>
<th>PSNR</th>
<td>29.64</td>
<td>28.85</td>
</tr><tr>
<th>SSIM</th>
<td>0.9257</td>
<td>0.9090</td>
</tr><tr>
<th>Result</th>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/ising_spins_salt_pepper_lenna.png"/></td>
<td><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/lenna/salt_pepper/median3_salt_pepper_lenna.png"/></td>
</tr></table>

Note that both the Ising model and median-3 results still contain some black-and-white patches. We intentionally applied strong salt-and-pepper noise (probability=0.07) to show this situation. Median-5 filtering can completely clean these patches but at the cost of blurring the image, which is why its PSNR and SSIM are lower. Another advantage of the Ising model over median-3 is that the simulated annealing process contains random variables like temperature-controlled flip probabilities. Thus, in some tests, the Ising model can also completely remove black-and-white patches, yielding better results—meaning the Ising model generally performs better.

### Real SPAD Image

#### Direct Normalization (Accumulating 16 grayscale frames)

```bat
python IsingModel.py -w 128 -h 128 -a exp -tmax 20 -tmin 1 -smin 0 -smax 255 -js c -jj tv-denoise\spad\system\normalize\j_16.txt -hl 4 -hh tv-denoise\spad\system\normalize\h_16.txt -e 10_000_000
```

<table><tr>
<th>Annealing Process</th>
<th>Energy Changes</th>
<th>Original Image</th>
<th>Processing Result</th>
</tr><tr>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/grayscale/16_anim.gif"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/grayscale/16_energy.png"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/noise/normalize/16.png"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/grayscale/16_spins.png"/></td>
</tr></table>

#### Reservoir Coding (Accumulating 3 frames & threshold >= 2)

```bat
python IsingModel.py -w 128 -h 128 -a exp -tmax 20 -tmin 1 -smin 0 -smax 1 -js c -jj tv-denoise\spad\system\j_n3t2.txt -hl 6 -hh tv-denoise\spad\system\h_n3t2.txt -e 1_000_000
```

<table><tr>
<th>Annealing Process</th>
<th>Energy Changes</th>
<th>Original Image</th>
<th>Processing Result</th>
</tr><tr>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/reservoir/n3t2_anim.gif"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/reservoir/n3t2_energy.png"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/noise/n3t2.png"/></td>
<td width=25%><img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/tv-denoise/spad/results/reservoir/n3t2_spins.png"/></td>
</tr></table>

## Parameter List

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

## Auxiliary Tools

### annealing/DrawCurves.py

Draw different annealing temperature curves.

```bat
cd annealing
python DrawCurves.py
```

<img src="https://media.githubusercontent.com/media/yiyaowen/IsingModel/refs/heads/main/annealing/curves.png" width=50%/>

### Convert.py

Reduce target problems (images or other files) to the Ising model (text files). For example, input `binary.png` to output `j_binary.txt` and `h_binary.txt` (optional). The resulting initial spin states and J-Coupling coefficients (in `j_binary.txt`) and H-Coupling coefficients (in `h_binary.txt`) can be used as inputs for `IsingModel.py`. Storing system parameters in text files facilitates manual modification and debugging, especially for quickly testing different initial states and coupling coefficients, at the cost of larger file sizes (negligible for modern computers).

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

Add noise to grayscale images. The following command generates `noise_raw.png`:

```bat
python AddNoise.py [-t salt_pepper] [-p 0.1] raw.png
```

| Long          | Short | Value                  |
|---------------|-------|------------------------|
| source        |       | path/to/raw_image      |
| --noise-type  | -t    | [salt_pepper, poisson] |
| --noise-param | -p    | 0.1                    |

### AddFilter.py

Apply filters to grayscale images. The following command applies a 3x3 median filter:

```bat
python AddFilter.py [-t median] [-p 1] noise.png
```

| Long           | Short | Value                         |
|----------------|-------|-------------------------------|
| source         |       | path/to/noise_image           |
| --filter-type  | -t    | [median, gaussian, bilateral] |
| --filter-param | -p    | 1                             |

### CalcMetrics.py

Calculate PSNR and SSIM metrics for noisy images relative to a base image:

```bat
python CalcPSNR.py <base_image> <noisy_image1> [<noisy_image2> ...]
```
