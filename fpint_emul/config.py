from dataclasses import dataclass


@dataclass(frozen=True)
class FpIntConfig:
    """Numerical contract for QCOL_REAL_2SCOMP linear emulation."""

    weight_bits: int
    group_size: int = 32
    mxu_rows: int = 32
    extra_bits: int = 19
    reduce_extra_bits: int = 10
    n_chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.weight_bits not in (4, 8):
            raise ValueError("weight_bits must be 4 or 8")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError("group_size must be -1 or a positive integer")
        if self.mxu_rows <= 0:
            raise ValueError("mxu_rows must be positive")
        if self.group_size > 0 and self.group_size % self.mxu_rows != 0:
            raise ValueError(
                "group_size must be a multiple of mxu_rows so a scale boundary "
                "does not split an MXU reduction tile"
            )
        if self.extra_bits < self.reduce_extra_bits:
            raise ValueError("extra_bits must be >= reduce_extra_bits")
        if self.extra_bits < 0 or self.reduce_extra_bits < 0:
            raise ValueError("extra bit widths must be non-negative")
        # The default has ample headroom: (11+19)-bit activations, an 8-bit
        # weight and a 32-lane sum remain below signed int64. Reject obviously
        # unsafe configurations before an integer tensor can wrap around.
        worst_main_bits = 11 + self.extra_bits + self.weight_bits + (
            self.mxu_rows - 1
        ).bit_length()
        if worst_main_bits > 53:
            raise ValueError(
                "configuration exceeds exact float64 integer contraction range"
            )
        if self.n_chunk_size <= 0:
            raise ValueError("n_chunk_size must be positive")

    def effective_group_size(self, k: int) -> int:
        if k <= 0:
            raise ValueError("K must be positive")
        return k if self.group_size == -1 else self.group_size

    def group_count(self, k: int) -> int:
        group_size = self.effective_group_size(k)
        return (k + group_size - 1) // group_size

    def tile_count(self, k: int) -> int:
        if k <= 0:
            raise ValueError("K must be positive")
        return (k + self.mxu_rows - 1) // self.mxu_rows

    def group_for_tile(self, tile: int, k: int) -> int:
        if self.group_size == -1:
            return 0
        return (tile * self.mxu_rows) // self.group_size

    def validate_zero_bound(self, maximum_absolute_zero: int) -> None:
        maximum_weight = 1 << (self.weight_bits - 1)
        maximum_mantissa = (1 << 11) - 1
        main = (
            (maximum_mantissa << self.extra_bits)
            * maximum_weight
            * self.mxu_rows
        )
        correction = (
            (maximum_mantissa << self.reduce_extra_bits)
            * self.mxu_rows
            * maximum_absolute_zero
            * (1 << (self.extra_bits - self.reduce_extra_bits))
        )
        if main + correction >= 1 << 63:
            raise ValueError("zero-point and configuration can overflow int64 post-processing")
