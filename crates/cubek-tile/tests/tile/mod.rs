mod arrival;
mod attention;
mod blocked;
mod census;
mod coarse;
mod conv;
mod decode_gemv;
mod depthwise;
mod dequant;
mod distributed;
mod erased;
mod imperative;
mod launcher;
mod leaf;
mod matmul;
mod packed;
mod paged;
mod procedural;
mod quant;
mod ragged;
mod recursive;
mod reduce;
mod references;
mod routed;
mod row_placement;
mod scaled;
mod separable;
mod smem_accumulation;
mod softmax;
mod space;
mod split_k;
mod stream;

use cubecl::prelude::*;
use cubek_tile::{Axis, Grid, Launcher, Partitioning, Space};

/// Which extents a test's kernel reads at runtime: the test's statement of what a family decides
/// once by building its partitioning over `Space::with_dynamic`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Form<'a> {
    Static,
    Dynamic,
    DynamicAlong(&'a [Axis]),
}

fn kernel_form(partitioning: &Partitioning, form: Form<'_>) -> Partitioning {
    let space = partitioning.space().clone();
    let kernel = match form {
        Form::Static => space,
        Form::Dynamic => space.all_dynamic(),
        Form::DynamicAlong(axes) => space.with_dynamic(axes),
    };
    Partitioning::new(kernel, partitioning.levels().to_vec())
}

/// A launch whose grid the levels imply, over the partitioning's own extents as the concrete ones.
pub(crate) fn implied(client: &Client, partitioning: Partitioning, form: Form<'_>) -> Launcher {
    let concrete = partitioning.space().clone();
    Launcher::new(
        client,
        kernel_form(&partitioning, form),
        &concrete,
        Grid::FromLevels,
    )
}

/// A launch of one cube of one unit over `kernel`, cut by no level, its dynamic axes sized off
/// `concrete`: what a kernel that reads the space and nothing else is handed.
pub(crate) fn uncut(client: &Client, kernel: &Space, concrete: &Space) -> Launcher {
    Launcher::new(
        client,
        Partitioning::new(kernel.clone(), Vec::new()),
        concrete,
        Grid::Stated {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
        },
    )
}
