use crate::{LineMode, ReduceDtypes, ReduceError, routines::ReduceBlueprint};
use cubecl::prelude::*;

#[derive(Debug)]
pub struct ReduceLineSettings {
    pub line_mode: LineMode,
    pub line_size_input: u8,
    pub line_size_output: u8,
}

#[derive(Debug)]
pub struct ReduceLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub line: ReduceLineSettings,
}

#[derive(Debug)]
pub struct ReduceProblem {
    pub vector_size: u32,
    pub vector_count: u32,
    pub axis: u32,
    pub dtypes: ReduceDtypes,
}

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    Forced(R::Blueprint, CubeDim),
    Inferred(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}

pub fn cube_count_safe<R: Runtime>(client: &ComputeClient<R>, num_cubes: u32) -> (CubeCount, u32) {
    let cube_count = cube_count_spread(&client.properties().hardware.max_cube_count, num_cubes);

    (
        CubeCount::Static(cube_count[0], cube_count[1], cube_count[2]),
        cube_count[0] * cube_count[1] * cube_count[2],
    )
}

fn cube_count_spread(max: &CubeCount, num_cubes: u32) -> [u32; 3] {
    let max_cube_counts = match max {
        CubeCount::Static(x, y, z) => [*x, *y, *z],
        CubeCount::Dynamic(_) => panic!("No static max cube count"),
    };
    let mut num_cubes = [num_cubes, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_cubes[i] <= max_cube_counts[i] {
            return true;
        }

        loop {
            num_cubes[i] = num_cubes[i].div_ceil(base);
            num_cubes[i + 1] *= base;

            if num_cubes[i] <= max_cube_counts[i] {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_cubes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_num_cubes_even() {
        let max = CubeCount::Static(32, 32, 32);
        let required = 2048;

        let actual = cube_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test]
    fn safe_num_cubes_odd() {
        let max = CubeCount::Static(48, 32, 16);
        let required = 3177;

        let actual = cube_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }
}
