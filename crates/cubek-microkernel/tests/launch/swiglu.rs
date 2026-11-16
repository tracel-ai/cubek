//! The SwiGLU microkernel on its work item: `rows` fused rows `[gate | up]`, each `dim` wide a
//! half, gated into `out = silu(gate) ⊙ up`, every cube one group running one row's run, written
//! straight into a plain output.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::{Launcher, Levels, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::row_store::{CubeStage, RowStore};
use cubek_microkernel::swiglu::cube::{
    CubeSwigluMicrokernel, CubeSwigluMicrokernelSettings, cube_swiglu_microkernel,
};
use cubek_microkernel::swiglu::{D, H, R, SwigluPartitioning, SwigluWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::RowBatch;

/// A SwiGLU microkernel ready to launch over `rows` fused rows.
#[derive(Debug, Clone, Copy)]
pub struct SwigluLaunch {
    pub microkernel: CubeSwigluMicrokernel,
    pub item: SwigluWorkItem,
    pub rows: usize,
}

impl SwigluLaunch {
    /// The cube SwiGLU microkernel `hw` runs at `settings` over `batch`'s rows of `element`, each
    /// half its `dim` wide.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        element: ElemType,
        batch: RowBatch,
        settings: &CubeSwigluMicrokernelSettings,
    ) -> Result<Self, Refusal> {
        let RowBatch { rows, dim } = batch;
        let item = SwigluWorkItem {
            element,
            dim,
            store: RowStore::Plain,
        };
        let microkernel = CubeSwigluMicrokernel::best(hw, &item, settings)?;
        Ok(Self {
            microkernel,
            item,
            rows,
        })
    }

    /// `out[r, d] = silu(input[r, 0, d]) · input[r, 1, d]`, `input` `[rows, 2, dim]` and `out`
    /// `[rows, dim]`, both row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(&self, client: &Client, input: TensorArg, out: TensorArg) -> Result<(), Refusal> {
        let store = self.item.store;
        let space = Space::new(
            &[
                store.rows_axes(self.rows),
                vec![(H, 2)],
                store.width_axes(self.item.dim),
            ]
            .concat(),
        );
        let run_box = [
            vec![(R, 1), (H, 2)],
            store.width_axes(self.microkernel.run()),
        ]
        .concat();
        let partitioning = SwigluPartitioning::new(
            space.clone(),
            Levels::leaf(&run_box).planes(&[(R, 1)]),
            store,
        )
        .as_tile()
        .clone();
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.microkernel.units as u32),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let (element, width) = (self.item.element, self.microkernel.width);
        // SAFETY: both operands are bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; a plain store stages nothing.
        unsafe {
            cube_swiglu_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                width,
                TileTensorLaunch::new(input, element, width, TileSpec::direct(&[R, H, D])),
                TileTensorLaunch::new(out, element, width, TileSpec::direct(&[R, D])),
                launcher.partitioning_arg(),
                self.microkernel,
                element,
            );
        }
        Ok(())
    }
}

/// Every cube running its group's run, written straight into the output.
#[cube(launch_unchecked)]
fn cube_swiglu_kernel<E: Float, N: Size>(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeSwigluMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let x = input.tile(&partitioning);
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        cube_swiglu_microkernel::<E, N>(
            &cube,
            &x.at(&cube),
            &o.at(&cube),
            microkernel,
            CubeStage::Direct,
        );
    }
}
