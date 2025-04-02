class Coordinate:
    """
    Help convert between Spin/J-coupling Coordinates and System Coordinates.
    For example, if we have a 3x3 lattice, then the system should be 5x5,
    in which case we have cd = Coordinate((3, 3)) and cd.sc(1, 2) -> (2, 4).
    """
    def __init__(self, width, height):
        # only include spins
        self.rows = height
        self.cols = width
        self.size = (self.rows, self.cols)

        # include spins and j_coupling
        self.full_rows = 2 * height - 1
        self.full_cols = 2 * width - 1
        self.full_size = (self.full_rows, self.full_cols)

    def sc(self, r, c):
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

    def jc(self, sc1, sc2):
        """
        J-coupling Coordinates
        Note that this is the coupling between two spins,
        and thus we need a pair of SCs to get the related J-coupling coordinates.
        """
        return (sc1[0] + sc2[0]) // 2, (sc1[1] + sc2[1]) // 2
