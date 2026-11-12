//! The cube staged settings the benchmarks and the tests pin, named by the grid of fragments a
//! plane holds, the planes of a cube, and the rows of the instruction.

use crate::kernels::tiled::ConvSettings;
use cubek_microkernel::matmul::cube::MatmulPlaneLayout;
use cubek_microkernel::matmul::plane::PlaneFragmentMicrokernelSettings as Fragments;

/// 2x2 fragments of 16 rows on each of 2x2 planes.
pub const F2X2_P2X2_T16: ConvSettings =
    ConvSettings::new(16, Fragments::grid(2, 2, 1), MatmulPlaneLayout::new(2, 2));

/// 1x2 fragments of 16 rows on each of 1x2 planes: one tile of rows.
pub const F1X2_P1X2_T16: ConvSettings =
    ConvSettings::new(16, Fragments::grid(1, 2, 1), MatmulPlaneLayout::new(1, 2));

/// 2x2 fragments of 16 rows on each of 1x2 planes: two tiles of rows.
pub const F2X2_P1X2_T16: ConvSettings =
    ConvSettings::new(16, Fragments::grid(2, 2, 1), MatmulPlaneLayout::new(1, 2));

/// 1x2 fragments of 16 rows on each of 2x2 planes: two tiles of rows.
pub const F1X2_P2X2_T16: ConvSettings =
    ConvSettings::new(16, Fragments::grid(1, 2, 1), MatmulPlaneLayout::new(2, 2));

/// Seven 8x32 instructions down each of two planes across the columns.
pub const F7X1_P1X2_T8: ConvSettings =
    ConvSettings::new(8, Fragments::grid(7, 1, 1), MatmulPlaneLayout::new(1, 2));

/// One 8x32 instruction on each of two planes across the columns.
pub const F1X1_P1X2_T8: ConvSettings =
    ConvSettings::new(8, Fragments::grid(1, 1, 1), MatmulPlaneLayout::new(1, 2));

/// One 8x32 plane for each of seven output rows, which share each staged weight block.
pub const F1X1_P1X1_T8_A7: ConvSettings =
    ConvSettings::new(8, Fragments::grid(1, 1, 1), MatmulPlaneLayout::new(1, 1)).across(7);
