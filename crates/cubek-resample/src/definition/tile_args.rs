use cubecl::{prelude::*, std::FastDivmod, zspace::Shape};

/// Tile args.
#[derive(CubeType, CubeLaunch)]
pub struct TileArgs {
    pub tile_shape: Sequence<FastDivmod<usize>>,
    pub tile_strides: Sequence<FastDivmod<usize>>,
    pub cube_shape: Sequence<FastDivmod<usize>>,
    pub cube_strides: Sequence<FastDivmod<usize>>,
    pub output_shape: Sequence<usize>,
}

/// Tile args launcher to convert to Sequence with FastDivmod and usize.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileArgsLauncher {
    pub tile_shape: Vec<usize>,
    pub tile_strides: Vec<usize>,
    pub cube_shape: Vec<usize>,
    pub cube_strides: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl TileArgsLauncher {
    /// Distributes the workload between threads in a tiled layout.
    pub fn new(
        output_shape: &Shape,
        cube_dim: &CubeDim,
        vectorized_axis: usize,
        vector_size: usize,
    ) -> TileArgsLauncher {
        let len = output_shape.len();

        let mut tile_shape = vec![1; len];
        let mut cube_shape = vec![1; len];

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

        let output_shape = output_shape.to_vec();

        TileArgsLauncher {
            tile_shape,
            tile_strides,
            cube_shape,
            cube_strides,
            output_shape,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tile_shape.is_empty()
    }

    pub fn num_cubes(&self) -> usize {
        self.cube_shape.iter().product()
    }

    pub fn to_launch<R: Runtime>(self) -> TileArgsLaunch<R> {
        TileArgsLaunch::new(
            to_sequence::<R, FastDivmod<usize>>(&self.tile_shape),
            to_sequence::<R, FastDivmod<usize>>(&self.tile_strides),
            to_sequence::<R, FastDivmod<usize>>(&self.cube_shape),
            to_sequence::<R, FastDivmod<usize>>(&self.cube_strides),
            to_sequence::<R, usize>(&self.output_shape),
        )
    }
}

/// Helper to compute row-major stride from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

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
