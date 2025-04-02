# IsingModel

Algorithms for Ising Model based SRAM-CIM Annealing Chip.

## Env Setup

```bat
pip install requirements.txt
```

## Examples

### Hello World

Simulate the phase transition process of crystals, which is the most original application scenario of the Ising model. By default, the J-coupling coefficients are all equal (e.g +1), so the final result often appears as clustered random color blocks.

```bat
python IsingModel.py
```

### Max-Cut Problem

Map the Max-Cut problem to the Ising model and then solve it using simulated annealing. This process is typically demonstrated by generating images, as the coloring of Max-Cut can directly correspond to different regions of a binary image.

#### Generate number 1

```bat
python IsingModel.py -w 28 -h 28 -tmax 20 -tmin 1 -jj j-coupling/number_1.txt
```

#### Generate word SEMI

```bat
python IsingModel.py -w 200 -h 100 -tmax 20 -tmin 1 -jj j-coupling/word_SEMI.txt -e 50_000_000
```

### Image Denoising

Map the Total Variation (TV) denoising algorithm to the Ising model. Although TV is a convex optimization problem (i.e., there exists a unique global minimum), and algorithms such as gradient descent can solve it well, the Ising model may have certain advantages in circuit implementation.

#### Denoise word ABC

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -js c -jj tv-denoise/ABC/j_noise_ABC.txt -hl 8.0 -hh tv-denoise/ABC/h_noise_ABC.txt
```

#### Denoise numbers

```bat
python IsingModel.py -w 64 -h 64 -a exp -tmax 20 -tmin 1 -js c -jj tv-denoise/numbers/system/j_noise_0.txt -hl 8.0 -hh tv-denoise/numbers/system/h_noise_0.txt
```

Replace `j_noise_0` and `h_noise_0` for different numbers.

## Argument Table

```bat
python IsingModel.py [arguments]
```

| Long           | Short | Value                |
|----------------|-------|----------------------|
| --width        | -w    | 28                   |
| --height       | -h    | 28                   |
| --graph-type   | -g    | [nearest, king]      |
| --annealing    | -a    | [lin, exp, cos] etc. |
| --anneal-temp1 | -tmax | 20                   |
| --anneal-temp2 | -tmin | 10                   |
| --init-state   | -js   | [r, u-1, u+1, c]     |
| --j-coupling   | -jj   | path/to/j_system.txt |
| --h-lambda     | -hl   | 1.0                  |
| --h-coupling   | -hh   | path/to/h_system.txt |
| --sample-count | -s    | 1                    |
| --epochs       | -e    | 1_000_000            |
| --output-video | -o    | result.mp4           |
| --frame-rate   | -f    | 5                    |
| --duration     | -d    | 3                    |
| --parallel     | -p    | 1                    |

## Helper Tools

### DrawCurves.py

Plot different annealing curves.

```bat
cd annealing
python DrawCurves.py
```

### Convert.py

Convert a problem into a system file (Ising Lattice) that can be loaded by IsingModel.

```bat
python Convert.py -p Generate input.png output.txt
```

| Long         | Short | Value                               |
|--------------|-------|-------------------------------------|
| --problem    | -p    | [Generate, Denoise]                 |
| --graph-type | -g    | [nearest, king] (only for Generate) |

### AddNoise.py

Add salt-and-pepper noise to a binary image. The result can be further converted to an Ising Lattice via `Convert.py -p Denoise [arguments]` as above.

```bat
cd tv-denoise
python AddNoise.py raw.png
```

This command will generate `noise_raw.png` in the same directory as `raw.png`.
