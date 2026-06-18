use cubecl::{
    prelude::*,
    std::{FastDivmod, FastDivmodExpand},
    zspace::{Shape, SmallVec},
};

/// Tile size.
#[derive(CubeType, CubeLaunch)]
pub struct TileSize {
    pub shape: Sequence<FastDivmod<usize>>,
    pub strides: Sequence<FastDivmod<usize>>,
}

#[cube]
impl TileSize {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn area(&self) -> usize {
        let mut area = 1;

        #[unroll]
        for i in 0..self.rank() {
            area *= fast_div_mod_value(&self.shape[i]);
        }

        area
    }
}

/// Tile size launcher to convert to Sequence with FastDivmod.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileSizeLauncher {
    pub shape: SmallVec<[usize; 8]>,
    pub strides: SmallVec<[usize; 8]>,
}

impl TileSizeLauncher {
    /// Distributes the workload between threads in a tiled layout.
    pub fn new(
        output_shape: &Shape,
        cube_dim: &CubeDim,
        vectorized_axis: usize,
        vector_size: usize,
    ) -> (TileSizeLauncher, TileSizeLauncher) {
        let len = output_shape.len();

        let mut tile_shape: SmallVec<[usize; 8]> = (0..len).map(|_| 1).collect();
        let mut cube_shape: SmallVec<[usize; 8]> = (0..len).map(|_| 1).collect();

        let mut remaining_cube_dim = cube_dim.num_elems() as usize;

        // Process dimensions in reverse order to ensure a cube processes contiguous memory (memory coalescing).
        for i in (0..len).rev() {
            let size = if vectorized_axis == i {
                output_shape[i] / vector_size
            } else {
                output_shape[i]
            };

            // This strategy ensure that the product of tile_sizes >= the original cube_dim.
            // Which guarantee that each thread will have at least one element to process.
            tile_shape[i] = size.min(remaining_cube_dim).max(1);
            cube_shape[i] = size.div_ceil(tile_shape[i]);

            remaining_cube_dim = remaining_cube_dim.div_ceil(tile_shape[i]);
        }

        let tile_strides = compute_strides(&tile_shape);
        let cube_strides = compute_strides(&cube_shape);

        (
            TileSizeLauncher {
                shape: tile_shape,
                strides: tile_strides,
            },
            TileSizeLauncher {
                shape: cube_shape,
                strides: cube_strides,
            },
        )
    }

    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn num_cubes(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_launch<R: Runtime>(&self) -> TileSizeLaunch<R> {
        TileSizeLaunch::new(
            to_sequence::<R, FastDivmod<usize>>(&self.shape),
            to_sequence::<R, FastDivmod<usize>>(&self.strides),
        )
    }
}

/// Get the value of a FastDivmod.
#[cube]
pub fn fast_div_mod_value(div_mod: &FastDivmod<usize>) -> usize {
    match div_mod {
        FastDivmod::Fast { divisor, .. } => *divisor,
        FastDivmod::Fallback { divisor } => *divisor,
    }
}

/// Helper to compute row-major stride from a shape.
fn compute_strides(shape: &[usize]) -> SmallVec<[usize; 8]> {
    let mut strides: SmallVec<[usize; 8]> = (0..shape.len()).map(|_| 1).collect();

    if shape.is_empty() {
        return strides;
    }

    // Iterate backwards starting from the second-to-last element
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

/// Convert a slice of dimensions into a `SequenceArg`.
fn to_sequence<R: Runtime, T: LaunchArg>(shape: &[usize]) -> SequenceArg<R, T>
where
    usize: Into<<T as LaunchArg>::RuntimeArg<R>>,
{
    let mut sequence = SequenceArg::new();
    for dim in shape.iter() {
        sequence.push((*dim).into());
    }
    sequence
}
