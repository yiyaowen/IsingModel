# 2D Ising Model Simulation with Metropolis Algorithm
# Copyleft (c) 2025 yiyaowen (wenyiyao23@semi.ac.cn)
#
# Code Repo: https://github.com/yiyaowen/IsingChip
# MIT License: https://opensource.org/licenses/MIT
#
# Preferably run this script with Python 3.10 or later


from functools import partial
from multiprocessing import get_context
from os import path
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

    Energy = -2J * S_i * S_j + L*H
    where S_i and S_j are the spins, J and L*H are the couplings

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
        init_state, j_coupling, h_lambda, h_coupling,
        epochs
        ):
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
        self.Tmax = anneal_temp1
        self.Tmin = anneal_temp2
        self.Tspan = self.Tmax - self.Tmin

        self.build_j_system(init_state, j_coupling)
        self.build_h_system(h_lambda, h_coupling)

        self.epochs = epochs

    @property
    def spins(self):
        return self.j_system[::2, ::2]

    def build_j_system(self, init_state, j_coupling):
        if j_coupling:
            self.j_system = np.loadtxt(j_coupling)
            if self.j_system.shape != self.full_size:
                raise ValueError("Invalid J-coupling Shape")
        else: # init with random spins and J-couplings
            self.j_system = np.random.choice([-1, 1], self.full_size)
        self.build_spins(init_state)
        self.build_coupling(j_coupling)

    def build_spins(self, init_state):
        match init_state:
            case "r":
                self.j_system[::2, ::2] = np.random.choice([-1, 1], self.size)
            case "u-1":
                self.j_system[::2, ::2] = -1 * np.ones(self.size)
            case "u+1":
                self.j_system[::2, ::2] = +1 * np.ones(self.size)
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
                        self.j_system[r, c] = +1
                else:
                    for c in range(0, self.full_cols, 1):
                        self.j_system[r, c] = +1

    def build_h_system(self, h_lambda, h_coupling):
        if h_coupling:
            self.h_system = h_lambda * np.loadtxt(h_coupling)
            if self.h_system.shape != self.size:
                raise ValueError("Invalid H-coupling Shape")
        else: # init with zero H-couplings
            self.h_system = np.zeros(self.size)

    def T(self, epoch):
        """
        Annealing Temperature
        For different annealing methods, the temperature changes differently.
        Add a new annealing method by defining a new case in the match statement.
        """
        x = epoch / self.epochs
        match self.annealing:
            case "lin":
                return self.Tmin + self.Tspan * (1 - x)
            case "exp":
                return self.Tmin + self.Tspan * np.exp(-5 * x)
            case "exp2":
                return self.Tmin + self.Tspan * np.exp(-10 * x)
            case "cos":
                return self.Tmin + self.Tspan * np.cos(np.pi/2 * x**2)
            case "cos2":
                return self.Tmin + self.Tspan * np.cos(np.pi/2 * x**4)
            case "plateau":
                return self.Tmin + self.Tspan * (0.5 - (1.6*x - 0.8)**3)
            case "sigmoid":
                return self.Tmin + self.Tspan * (1 - 1 / (1 + np.exp(5 - 10 * x)))
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
            _r = 0
        elif r >= self.rows:
            _r = self.rows - 1

        _c = c
        if c < 0:
            _c = 0
        elif c >= self.cols:
            _c = self.cols - 1

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
        Ei = Sum(-2J * S_i * S_j) + L*H
        where S_i/j is the coupled spins.
        """
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
                sc0 = self._sc(r + 0, c + 0)
                sc1 = self._sc(r - 1, c + 0)
                sc2 = self._sc(r + 0, c + 1)
                sc3 = self._sc(r + 1, c + 0)
                sc4 = self._sc(r + 0, c - 1)
                return -2 * (
                    self.j_system[self._jc(sc0, sc1)] * self.j_system[sc0] * self.j_system[sc1] + # 1
                    self.j_system[self._jc(sc0, sc2)] * self.j_system[sc0] * self.j_system[sc2] + # 2
                    self.j_system[self._jc(sc0, sc3)] * self.j_system[sc0] * self.j_system[sc3] + # 3
                    self.j_system[self._jc(sc0, sc4)] * self.j_system[sc0] * self.j_system[sc4]   # 4
                    ) - self.h_system[r, c] * self.spins[r, c] # L*H
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
                sc0 = self._sc(r + 0, c + 0)
                sc1 = self._sc(r - 1, c - 1)
                sc2 = self._sc(r - 1, c + 0)
                sc3 = self._sc(r - 1, c + 1)
                sc4 = self._sc(r + 0, c + 1)
                sc5 = self._sc(r + 1, c + 1)
                sc6 = self._sc(r + 1, c + 0)
                sc7 = self._sc(r + 1, c - 1)
                sc8 = self._sc(r + 0, c - 1)
                return -2 * (
                    self.j_system[self._jc(sc0, sc1)] * self.j_system[sc0] * self.j_system[sc1] + # 1
                    self.j_system[self._jc(sc0, sc2)] * self.j_system[sc0] * self.j_system[sc2] + # 2
                    self.j_system[self._jc(sc0, sc3)] * self.j_system[sc0] * self.j_system[sc3] + # 3
                    self.j_system[self._jc(sc0, sc4)] * self.j_system[sc0] * self.j_system[sc4] + # 4
                    self.j_system[self._jc(sc0, sc5)] * self.j_system[sc0] * self.j_system[sc5] + # 5
                    self.j_system[self._jc(sc0, sc6)] * self.j_system[sc0] * self.j_system[sc6] + # 6
                    self.j_system[self._jc(sc0, sc7)] * self.j_system[sc0] * self.j_system[sc7] + # 7
                    self.j_system[self._jc(sc0, sc8)] * self.j_system[sc0] * self.j_system[sc8]   # 8
                    ) - self.h_system[r, c] * self.spins[r, c] # L*H
            case _:
                raise ValueError("Invalid Graph Type")

    @property
    def internal_energy(self):
        e = 0
        E = 0
        E_2 = 0

        for r in range(self.rows):
            for c in range(self.cols):
                e = self.energy(r, c)
                E += e
                E_2 += e**2

        U = (1.0 / self.count) * E
        U_2 = (1.0 / self.count) * E_2

        return U, U_2

    @property
    def magnetization(self):
        return np.abs(np.sum(self.spins) / self.count)

    @property
    def heat_capacity(self):
        U, U_2 = self.internal_energy
        return U_2 - U**2


def run(lattice, sample_count, epochs, output_video, frame_rate, duration, n):
    """
    Run the Metropolis simulation and save the output video (*.mp4).
    """
    plt.ion()
    fig = plt.figure()

    FFMpegWriter = anim.writers["ffmpeg"]
    writer = FFMpegWriter(fps=frame_rate)

    frame_count = frame_rate * duration
    record_point = epochs // frame_count

    with writer.saving(fig, output_video, dpi=192):
        for i in tqdm(range(epochs), desc=f"#{n:3}", position=n, leave=False):

            #########################################
            # Randomly select a site on the lattice #
            #########################################

            cds = [np.random.randint(0, lattice.size, 2)
                   for i in range(sample_count)]

            ##########################################
            # Calculate the energy of a flipped spin #
            ##########################################

            E = [-1 * lattice.energy(*cd) for cd in cds]

            ###############################################
            # Roll the dice to see if the spin is flipped #
            ###############################################

            T = lattice.T(i)
            for cd, e in zip(cds, E):
                r, c = cd
                if e <= 0.0:
                    lattice.spins[r, c] *= -1
                else: # e > 0.0
                    p = np.exp(-e / T)
                    if np.random.rand() <= p:
                        lattice.spins[r, c] *= -1

            ########################################
            # Save the frame for each record point #
            ########################################

            if i % (record_point) == 0:
                img = plt.imshow(
                    lattice.spins,
                    cmap="tab20", # blue color map
                    interpolation="nearest" # pixel style
                    )
                writer.grab_frame() # encode the frame
                img.remove() # prepare for next frame

    plt.close("all")


def fun(
    width, height, graph_type,
    annealing, anneal_temp1, anneal_temp2,
    init_state, j_coupling, h_lambda, h_coupling,
    sample_count, epochs,
    output_video, frame_rate, duration,
    n, # subprocess No
    ):
    lattice = IsingLattice(
        width=width, height=height, graph_type=graph_type,
        annealing=annealing, anneal_temp1=anneal_temp1, anneal_temp2=anneal_temp2,
        init_state=init_state, j_coupling=j_coupling, h_lambda=h_lambda, h_coupling=h_coupling,
        epochs=epochs
        )
    # Test the lattice build
    # np.set_printoptions(threshold=np.inf)
    # print(lattice.system)
    # print(lattice.spins)
    # exit()
    output_video = f"{path.splitext(output_video)[0]}_{n}.mp4"
    run(lattice, sample_count, epochs, output_video, frame_rate, duration, n)

    # print(f"{'Magnetization [%]:':.<25}{lattice.magnetization:.2f}")
    # print(f"{'Heat Capacity [AU]:':.<25}{lattice.heat_capacity:.2f}")


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
    default=0.5,
    help="Maximal temperature during annealing"
    )
@click.option(
    "--anneal-temp2", "-tmin",
    default=0.5,
    help="Minimal temperature during annealing"
    )
@click.option(
    "--init-state", "-js",
    default="r",
    type=click.Choice(["r", "u-1", "u+1", "c"]),
    help="(r)andom\n(u)niform -1\n(u)niform +1\n(c)ustom"
    )
@click.option(
    "--j-coupling", "-jj",
    default=None, type=str,
    help="Set custom J-coupling input (*.txt)"
    # All J-couplings are set to +1 by default.
    # Note that the file should include both the init-spin state and the J-couplings state.
    # For example, if the lattice is 3x3, the content should be a 5x5 matrix (2n-1 for n).
    # In addition, you also need to set --init-state to 'c' to enable the custom init state,
    # otherwise the loaded init-spin state will be overwritten by the --init-state option.
    )
@click.option(
    "--h-lambda", "-hl",
    default=1.0,
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
    default=1_000_000, type=int,
    help="Number of epochs to run the simulation"
    # An experience value is 1_000_000 for a 100x100 lattice, or more for a larger lattice.
    )
@click.option(
    "--output-video", "-o",
    default="result.mp4",
    help="Output video file name (*.mp4)"
    )
@click.option(
    "--frame-rate", "-f",
    default=5, type=int,
    help="Frame rate of the output video"
    )
@click.option(
    "--duration", "-d",
    default=3, type=int,
    help="Duration of the output video in seconds"
    )
@click.option(
    "--parallel", "-p",
    default=1, type=int,
    help="Parallel count of the simulation"
    )
def main(
    width, height, graph_type,
    annealing, anneal_temp1, anneal_temp2,
    init_state, j_coupling, h_lambda, h_coupling,
    sample_count, epochs,
    output_video, frame_rate, duration,
    parallel
    ):
    t1 = time()

    ctx = get_context("spawn")
    lock = ctx.RLock()
    tqdm.set_lock(lock)

    job = partial(
        fun, width, height, graph_type,
        annealing, anneal_temp1, anneal_temp2,
        init_state, j_coupling, h_lambda, h_coupling,
        sample_count, epochs,
        output_video, frame_rate, duration
        )
    L = list(range(parallel))

    with ctx.Pool(initializer=tqdm.set_lock, initargs=(lock,)) as p:
        p.map(job, L)

    t2 = time()
    elapsed = round(t2 - t1)
    seconds = elapsed % 60
    minutes = elapsed // 60
    cprint(f"\n{parallel} finished with {minutes} minutes {seconds} seconds\n", "green")

if __name__ == "__main__":
    main()
