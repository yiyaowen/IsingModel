# 2D Ising Model Simulation with Metropolis Algorithm
# Copyleft (c) 2025 yiyaowen (wenyiyao23@semi.ac.cn)
#
# Code Repo: https://github.com/yiyaowen/IsingModel
# MIT License: https://opensource.org/licenses/MIT
#
# Preferably run this script with Python 3.10 or later


from functools import partial
from multiprocessing import get_context
from os import path
from PIL import Image
from termcolor import cprint
from time import time
from tqdm import tqdm
import click
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np


class IsingLattice:
    """
    2D Ising Model Lattice

    Nearest Graph       King's Graph
    -------------       -------------
    |   | 1 |   |       | 1 | 2 | 3 |
    ------↑------       ----↖-↑-↗----
    | 4 ← 0 → 2 |       | 8 ← 0 → 4 |
    ------↓------       ----↙-↓-↘----
    |   | 3 |   |       | 7 | 6 | 5 |
    -------------       -------------

    Energy = J * | S_i - S_j | + L * | S_i/j - H |
    where S_i and S_j are the spins, J, L and H are the couplings

    Metropolis Algorithm:
    1. Randomly select a site on the lattice.
    2. Calculate the energy of a flipped spin.
    3. Roll the dice to see if the spin is flipped.
    4. Repeat the process for a number of epochs.

    Flip probability = exp(-E / T)
    where E is the spin's local Hamiltonian, T is the temperature.
    """
    def __init__(
        self,
        width, height, graph_type,
        annealing, anneal_temp1, anneal_temp2,
        init_state, spin_space0, spin_space1,
        j_coupling, h_lambda, h_coupling, epochs
        ):
        if width <= 1 or height <= 1:
            raise ValueError("Invalid: width or height <= 1")

        # only include spins
        self.rows = height
        self.cols = width
        self.size = (self.rows, self.cols)
        self.count = self.rows * self.cols
        self.graph_type = graph_type

        # include spins and coupling
        self.full_rows = 2 * height - 1
        self.full_cols = 2 * width - 1
        self.full_size = (self.full_rows, self.full_cols)
        self.full_count = self.full_rows * self.full_cols

        self.annealing = annealing
        self.tmax = anneal_temp1
        self.tmin = anneal_temp2
        if self.tmax < self.tmin:
            raise ValueError("Invalid: tmax < tmin")
        self.tspan = self.tmax - self.tmin

        self.smin = spin_space0
        self.smax = spin_space1
        if self.smin > self.smax:
            raise ValueError("Invalid: smin > smax")
        self.sspan = self.smax - self.smin

        self.build_j_system(init_state, j_coupling)
        self.build_h_system(h_lambda, h_coupling)

        self.epochs = epochs
        self.energy_norm = 1 / (self.count * self.sspan)

    @property
    def spins(self):
        return self.j_system[::2, ::2]

    def random_s(self, size=None):
        return np.random.randint(self.smin, self.smax + 1, size=size)

    def build_j_system(self, init_state, j_coupling):
        if j_coupling:
            self.j_system = np.loadtxt(j_coupling, dtype=int)
            if self.j_system.shape != self.full_size:
                raise ValueError("Invalid J-coupling Shape")
        else: # init with random spins and J-couplings
            self.j_system = self.random_s(self.full_size)
        self.build_spins(init_state)
        self.build_coupling(j_coupling)

    def build_spins(self, init_state):
        match init_state:
            case "r":
                self.j_system[::2, ::2] = self.random_s(self.size)
            case "u0":
                self.j_system[::2, ::2] = np.full(self.size, self.smin)
            case "u1":
                self.j_system[::2, ::2] = np.full(self.size, self.smax)
            case "c":
                pass # the spins are already set in the system
            case _:
                raise ValueError("Invalid Init State")

    def build_coupling(self, j_coupling):
        if j_coupling is None:
            # TODO: Optimize the reset of the J-couplings
            for r in range(self.full_rows):
                if r % 2 == 0:
                    for c in range(1, self.full_cols, 2):
                        self.j_system[r, c] = 1
                else:
                    for c in range(0, self.full_cols, 1):
                        self.j_system[r, c] = 1

    def build_h_system(self, h_lambda, h_coupling):
        self.h_lambda = h_lambda
        if h_coupling:
            self.h_system = np.loadtxt(h_coupling, dtype=int)
            if self.h_system.shape != self.size:
                raise ValueError("Invalid H-coupling Shape")
        else: # init with zero H-couplings
            self.h_system = np.zeros(self.size, dtype=int)

    def T(self, epoch):
        """
        Annealing Temperature
        For different annealing methods, the temperature changes differently.
        Add a new annealing method by defining a new case in the match statement.
        """
        x = epoch / self.epochs
        match self.annealing:
            case "lin":
                return self.tmin + self.tspan * (1 - x)
            case "exp":
                return self.tmin + self.tspan * np.exp(-5 * x)
            case "exp2":
                return self.tmin + self.tspan * np.exp(-10 * x)
            case "cos":
                return self.tmin + self.tspan * np.cos(np.pi/2 * x**2)
            case "cos2":
                return self.tmin + self.tspan * np.cos(np.pi/2 * x**4)
            case "plateau":
                return self.tmin + self.tspan * (0.5 - (1.6*x - 0.8)**3)
            case "sigmoid":
                return self.tmin + self.tspan * (1 - 1 / (1 + np.exp(5 - 10 * x)))
            case _:
                raise ValueError("Invalid Annealing Method")

    def _sc(self, r, c):
        """
        Spin Coordinates
        As we maintain a full lattice of both spins and J-couplings,
        we need to map the spin coordinates to the full lattice coordinates.
        """
        _r = r
        if r < 0:
            _r = 1
        elif r >= self.rows:
            _r = self.rows - 2

        _c = c
        if c < 0:
            _c = 1
        elif c >= self.cols:
            _c = self.cols - 2

        return 2*_r, 2*_c

    def _jc(self, sc1, sc2):
        """
        J-coupling Coordinates
        Note that this is the coupling between two spins,
        and thus we need a pair of SCs to get the related J-coupling coordinates.
        """
        return (sc1[0] + sc2[0]) // 2, (sc1[1] + sc2[1]) // 2

    def energy(self, r, c):
        """
        Spin's Local Hamiltonian
        Ei/j = Sum(J * | S_i - S_j |) + L * | S_i/j - H |
        where S_i/j is the coupled spins.
        """
        return self.j_energy(r, c) + self.h_energy(r, c)

    def j_energy(self, r, c, spin_value=None):
        """
        Sum(J * | S_i - S_j |)
        """
        sc0 = self._sc(r + 0, c + 0)
        if spin_value is None:
            spin_value = self.j_system[sc0]

        match self.graph_type:
            case "nearest":
                """
                Nearest Graph
                -------------
                |   | 1 |   |
                ------↑------
                | 4 ← 0 → 2 |
                ------↓------
                |   | 3 |   |
                -------------
                """
                sc1 = self._sc(r - 1, c + 0)
                sc2 = self._sc(r + 0, c + 1)
                sc3 = self._sc(r + 1, c + 0)
                sc4 = self._sc(r + 0, c - 1)
                return (
                    self.j_system[self._jc(sc0, sc1)] * abs(spin_value - self.j_system[sc1]) + # 1
                    self.j_system[self._jc(sc0, sc2)] * abs(spin_value - self.j_system[sc2]) + # 2
                    self.j_system[self._jc(sc0, sc3)] * abs(spin_value - self.j_system[sc3]) + # 3
                    self.j_system[self._jc(sc0, sc4)] * abs(spin_value - self.j_system[sc4])   # 4
                    )
            case "king":
                """
                King's Graph
                -------------
                | 1 | 2 | 3 |
                ----↖-↑-↗----
                | 8 ← 0 → 4 |
                ----↙-↓-↘----
                | 7 | 6 | 5 |
                -------------
                """
                sc1 = self._sc(r - 1, c - 1)
                sc2 = self._sc(r - 1, c + 0)
                sc3 = self._sc(r - 1, c + 1)
                sc4 = self._sc(r + 0, c + 1)
                sc5 = self._sc(r + 1, c + 1)
                sc6 = self._sc(r + 1, c + 0)
                sc7 = self._sc(r + 1, c - 1)
                sc8 = self._sc(r + 0, c - 1)
                return (
                    self.j_system[self._jc(sc0, sc1)] * abs(spin_value - self.j_system[sc1]) + # 1
                    self.j_system[self._jc(sc0, sc2)] * abs(spin_value - self.j_system[sc2]) + # 2
                    self.j_system[self._jc(sc0, sc3)] * abs(spin_value - self.j_system[sc3]) + # 3
                    self.j_system[self._jc(sc0, sc4)] * abs(spin_value - self.j_system[sc4]) + # 4
                    self.j_system[self._jc(sc0, sc5)] * abs(spin_value - self.j_system[sc5]) + # 5
                    self.j_system[self._jc(sc0, sc6)] * abs(spin_value - self.j_system[sc6]) + # 6
                    self.j_system[self._jc(sc0, sc7)] * abs(spin_value - self.j_system[sc7]) + # 7
                    self.j_system[self._jc(sc0, sc8)] * abs(spin_value - self.j_system[sc8])   # 8
                    )
            case _:
                raise ValueError("Invalid Graph Type")

    def h_energy(self, r, c, spin_value=None):
        """
        L * | S_i/j - H |
        """
        if spin_value is None:
            spin_value = self.spins[r, c]

        return self.h_lambda * abs(spin_value - self.h_system[r, c])

    @property
    def total_energy(self):
        e = 0
        E = 0

        for r in range(self.rows):
            for c in range(self.cols):
                e = self.energy(r, c)
                E += e

        return E * self.energy_norm


def run(
    lattice, sample_count, epochs,
    output_path, frame_scale, frame_rate, anim_duration,
    n # subprocess No
    ):
    """
    Run the Metropolis simulation and save the output frames (*.gif/mp4).
    """
    plt.ion()
    output_info = path.splitext(output_path)

    writer = None
    match output_info[1]:
        case ".gif":
            writer = anim.PillowWriter(fps=frame_rate)
        case ".mp4":
            FFMpegWriter = anim.writers["ffmpeg"]
            writer = FFMpegWriter(fps=frame_rate)
        case _:
            raise ValueError("Invalid Output File Format")

    scale = frame_scale / 100
    figsize = (lattice.cols*scale, lattice.rows*scale)
    fig = plt.figure(num=1, figsize=figsize, dpi=100)

    anim_path = None
	# Output GIF file by default
    if output_info[1] == "":
        anim_path = f"{output_info[0]}{n}_anim.gif"
    else: # append the subprocess No to the output file name
        anim_path = f"{output_info[0]}{n}_anim{output_info[1]}"

    frame_count = frame_rate * anim_duration
    record_point = max(epochs // frame_count, 1)

    init_energy = lattice.total_energy
    epoch_detail = []
    energy_detail = []

    with writer.saving(fig, anim_path, dpi=100):
        for i in tqdm(range(epochs), desc=f"#{n:3}", position=n, leave=False):

            #########################################
            # Randomly select a site on the lattice #
            #########################################

            cds = [np.random.randint(
                (0, 0), lattice.size)
                for _ in range(sample_count)]

            ##########################################
            # Calculate the energy of a flipped spin #
            ##########################################

            Ej = [] # J-coupling energy before flip
            Eh = [] # H-coupling energy before flip
            flipped = [] # flipped spin values
            Ej_f = [] # J-coupling energy after flip
            Eh_f = [] # H-coupling energy after flip

            for cd in cds:
                Ej.append(lattice.j_energy(*cd))
                Eh.append(lattice.h_energy(*cd))
                f = lattice.random_s()
                flipped.append(f)
                Ej_f.append(lattice.j_energy(*cd, f))
                Eh_f.append(lattice.h_energy(*cd, f))

            ###############################################
            # Roll the dice to see if the spin is flipped #
            ###############################################

            T = lattice.T(i)
            params = zip(cds, Ej, Eh, flipped, Ej_f, Eh_f)

            for cd, ej, eh, f, ej_f, eh_f in params:
                e = ej + eh
                e_f = ej_f + eh_f
                if e_f <= e or np.random.rand() <= np.exp(-e_f / T):
                    r, c = cd
                    lattice.spins[r, c] = f
                    e_delta = 2 * (ej_f + eh_f) - (ej + eh)
                    init_energy += e_delta * lattice.energy_norm

            ########################################
            # Save the frame for each record point #
            ########################################

            if i % (record_point) == 0 or i == epochs - 1:
                img = plt.imshow(
                    lattice.spins,
                    vmin=lattice.smin,
                    vmax=lattice.smax,
                    cmap="gray", # grayscale
                    interpolation="nearest" # pixel style
                    )
                plt.axis("off")
                plt.subplots_adjust(0, 0, 1, 1)
                writer.grab_frame() # encode the frame
                img.remove() # prepare for next frame
                epoch_detail.append(i)
                energy_detail.append(init_energy)

    plt.close("all")

    # Save final spin-states
    spins = (lattice.spins - lattice.smin) / lattice.sspan * 255
    spins_image = Image.fromarray(spins.astype(np.uint8), mode="L")
    spins_image.save(f"{output_info[0]}{n}_spins.png")

    # Save energy-epoch curve
    plt.figure()
    plt.plot(epoch_detail, energy_detail)
    plt.title("Energy")
    plt.xlabel("Epochs")
    plt.ylabel("Normalized")
    plt.savefig(f"{output_info[0]}{n}_energy.png")
    plt.close("all")


def fun(
    width, height, graph_type,
    annealing, anneal_temp1, anneal_temp2,
    init_state, spin_space0, spin_space1, j_coupling,
    h_lambda, h_coupling, sample_count, epochs,
    output_path, frame_scale, frame_rate, anim_duration,
    n, # subprocess No
    ):
    lattice = IsingLattice(
        width=width, height=height, graph_type=graph_type,
        annealing=annealing, anneal_temp1=anneal_temp1, anneal_temp2=anneal_temp2,
        init_state=init_state, spin_space0=spin_space0, spin_space1=spin_space1,
        j_coupling=j_coupling, h_lambda=h_lambda, h_coupling=h_coupling, epochs=epochs
        )
    # Test the lattice build
    # np.set_printoptions(threshold=np.inf)
    # print(lattice.spins)
    # print(lattice.j_system)
    # print(lattice.h_system)

    run(lattice, sample_count, epochs, output_path, frame_scale, frame_rate, anim_duration, n)


@click.command()
@click.option(
    "--width", "-w",
    default=100,
    help="Width of spins in the Ising lattice"
    )
@click.option(
    "--height", "-h",
    default=100,
    help="Height of spins in the Ising lattice"
    )
@click.option(
    "--graph-type", "-g",
    default="king",
    type=click.Choice(["nearest", "king"]),
    help="Graph type (topology) of the spins"
    )
@click.option(
    "--annealing", "-a",
    default="lin",
    type=click.Choice(["lin", "exp", "exp2", "cos", "cos2", "plateau", "sigmoid"]),
    help="Choose the annealing method"
    )
@click.option(
    "--anneal-temp1", "-tmax",
    default=0.002,
    help="Maximal temperature during annealing"
    )
@click.option(
    "--anneal-temp2", "-tmin",
    default=0.002,
    help="Minimal temperature during annealing"
    )
@click.option(
    "--init-state", "-js",
    default="r",
    type=click.Choice(["r", "u0", "u1", "c"]),
    help="(r)andom\n(u)niform 0\n(u)niform 1\n(c)ustom"
    )
@click.option(
    "--spin-space0", "-smin",
    default=0,
    help="Minimal spin value in the lattice"
    )
@click.option(
    "--spin-space1", "-smax",
    default=255,
    help="Maximal spin value in the lattice"
    )
@click.option(
    "--j-coupling", "-jj",
    default=None, type=str,
    help="Set custom J-coupling input (*.txt)"
    # All J-couplings are set to 1 by default.
    # Note that the file should include both the init-spin state and the J-couplings state.
    # For example, if the lattice is 3x3, the content should be a 5x5 matrix (2n-1 for n).
    # In addition, you also need to set --init-state to 'c' to enable the custom init state,
    # otherwise the loaded init-spin state will be overwritten by the --init-state option.
    )
@click.option(
    "--h-lambda", "-hl",
    default=0,
    help="Set custom H-lambda for H-coupling"
    )
@click.option(
    "--h-coupling", "-hh",
    default=None, type=str,
    help="Set custom H-coupling input (*.txt)"
    )
@click.option(
    "--sample-count", "-c",
    default=1,
    help="Sample count of flip-spins in each epoch"
    )
@click.option(
    "--epochs", "-e",
    default=1_000_000,
    help="Number of epochs to run the simulation"
    # An experience value is 1_000_000 for a 100x100 lattice, or more for a larger lattice.
    )
@click.option(
    "--output-path", "-o",
    default="output-files/a.gif",
    help="Output file name (*.gif/mp4)"
    )
@click.option(
    "--frame-scale", "-s",
    default=1,
    help="Frame scale of the output file"
    )
@click.option(
    "--frame-rate", "-f",
    default=5,
    help="Frame rate of the output file"
    )
@click.option(
    "--anim-duration", "-d",
    default=3,
    help="Duration of the animation in seconds"
    )
@click.option(
    "--parallel-count", "-p",
    default=1,
    help="Parallel count of the simulation"
    )
def main(
    width, height, graph_type,
    annealing, anneal_temp1, anneal_temp2,
    init_state, spin_space0, spin_space1, j_coupling,
    h_lambda, h_coupling, sample_count, epochs,
    output_path, frame_scale, frame_rate, anim_duration,
    parallel_count
    ):
    t1 = time()

    ctx = get_context("spawn")
    lock = ctx.RLock()
    tqdm.set_lock(lock)

    job = partial(
        fun, width, height, graph_type,
        annealing, anneal_temp1, anneal_temp2,
        init_state, spin_space0, spin_space1, j_coupling,
        h_lambda, h_coupling, sample_count, epochs,
        output_path, frame_scale, frame_rate, anim_duration
        )
    L = list(range(parallel_count))

    with ctx.Pool(initializer=tqdm.set_lock, initargs=(lock,)) as p:
        p.map(job, L)

    t2 = time()
    elapsed = round(t2 - t1)
    seconds = elapsed % 60
    minutes = elapsed // 60
    cprint(f"\n{parallel_count} finished with {minutes} minutes {seconds} seconds\n", "green")

if __name__ == "__main__":
    main()
