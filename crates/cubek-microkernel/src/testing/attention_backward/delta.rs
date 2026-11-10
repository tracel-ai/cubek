//! The plane delta microkernel on its work item: one plane's queries of one head, one a unit,
//! each query's `delta = Σ_dv out_gradient · out`.

use crate::attention::{B, DV, G, KV, Q};
use crate::attention_backward::AttentionBackwardWorkItem;
use crate::attention_backward::delta::plane::{
    PlaneDeltaMicrokernel, PlaneDeltaMicrokernelSettings, plane_delta_microkernel,
};
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Launcher, Partitioning, Space, TileSpec, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// A plane delta microkernel ready to launch over one plane's queries.
#[derive(Debug, Clone, Copy)]
pub struct PlaneDeltaLaunch {
    pub microkernel: PlaneDeltaMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub plane_width: u32,
}

impl PlaneDeltaLaunch {
    /// The plane delta microkernel `hw` runs at `settings` over outputs of `element`, `val_dim`
    /// wide, summed in `sum`.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        element: ElemType,
        sum: ElemType,
        val_dim: usize,
        settings: &PlaneDeltaMicrokernelSettings,
    ) -> Result<Self, Refusal> {
        let item = AttentionBackwardWorkItem {
            element,
            sum,
            group: 1,
            head_dim: val_dim,
            val_dim,
        };
        let microkernel = PlaneDeltaMicrokernel::best(hw, &item, settings)?;
        Ok(Self {
            microkernel,
            item,
            plane_width: hw.plane_dim()?,
        })
    }

    /// The queries one plane holds.
    pub fn queries(&self) -> usize {
        self.microkernel.settings.seq_q.get()
    }

    /// `delta` `[1, 1, 1, queries]` in the sum's element, from `out` and `out_gradient`
    /// `[1, 1, 1, queries, val_dim]`, all row-major.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    pub fn launch(
        &self,
        client: &Client,
        out: TensorArg,
        out_gradient: TensorArg,
        delta: TensorArg,
    ) -> Result<(), Refusal> {
        let space = Space::new(&[
            (B, 1),
            (KV, 1),
            (G, 1),
            (Q, self.queries()),
            (DV, self.item.val_dim),
        ]);
        let partitioning = Partitioning::new(
            space.clone(),
            self.microkernel
                .tiling()
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, 1)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |arg, element, axes: &[_]| {
            TileTensorLaunch::new(arg, element, 1, TileSpec::direct(axes))
        };
        let (element, sum) = (self.item.element, self.item.sum);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            plane_delta_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                bound(out, element, &[B, KV, G, Q, DV]),
                bound(out_gradient, element, &[B, KV, G, Q, DV]),
                bound(delta, sum, &[B, KV, G, Q]),
                launcher.partitioning_arg(),
                sum,
            );
        }
        Ok(())
    }
}

/// The one plane writing its queries' dot products of the output with its gradient.
#[cube(launch_unchecked)]
fn plane_delta_kernel<EA: Float>(
    out: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    partitioning: Partitioning,
    #[define(EA)] _sum: ElemType,
) {
    let (o, d_o) = (out.tile(&partitioning), out_gradient.tile(&partitioning));
    let delta = delta.tile(&partitioning);
    for plane in &partitioning {
        plane_delta_microkernel::<EA>(&plane, &delta.at(&plane), &d_o.at(&plane), &o.at(&plane));
    }
}
