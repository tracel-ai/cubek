//! The unit matmul microkernel on one unit's block of the output, the cube's only unit, walking `K`
//! a step of its block at a time.

use crate::matmul::unit::{
    UnitRegistersMicrokernel, UnitRegistersMicrokernelSettings, unit_registers_microkernel,
};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{ComputeScope, UnitCube};
use cubecl::tile::{Partitioning, Semiring, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{M, N};
use cubek_std::microkernel::Microkernel;

use crate::matmul::MatmulMicrokernel;

use super::{MatmulElements, MatmulLaunch, OneCube};

/// One unit registers microkernel ready to launch on one unit's block, `steps` steps of `K` deep.
#[derive(Debug, Clone)]
pub struct UnitRegistersLaunch {
    pub microkernel: UnitRegistersMicrokernel,
    pub elements: MatmulElements,
    one_cube: OneCube,
}

impl UnitRegistersLaunch {
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: MatmulElements,
        settings: &UnitRegistersMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = UnitRegistersMicrokernel::best(hw, &elements.work_item(), settings)?;
        let levels = microkernel
            .tiling()
            .launch_units(
                &[M, N],
                UnitCube {
                    plane_width: 1,
                    planes: 1,
                },
            )
            .build();
        Ok(Self {
            microkernel,
            elements,
            one_cube: OneCube::new(levels, ComputeScope::Unit, steps),
        })
    }
}

impl MatmulLaunch for UnitRegistersLaunch {
    fn elements(&self) -> MatmulElements {
        self.elements
    }

    fn shape(&self) -> [usize; 3] {
        self.one_cube.shape
    }

    fn launch(
        &self,
        client: &Client,
        lhs: TensorArg,
        rhs: TensorArg,
        out: TensorArg,
    ) -> Result<(), Refusal> {
        let launcher = self.one_cube.launcher(client, 1)?;
        let [lhs, rhs, out] = self.elements.bound([lhs, rhs, out], self.rhs_layout());
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; it holds no shared memory.
        unsafe {
            unit_registers_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                lhs,
                rhs,
                out,
                launcher.partitioning_arg(),
                self.microkernel,
                [self.elements.sum, self.elements.out],
            );
        }
        Ok(())
    }
}

/// Every unit summing its block, walked within the rhs, and draining it into `out`.
#[cube(launch_unchecked)]
fn unit_registers_kernel<EA: Numeric, E: Numeric>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitRegistersMicrokernel,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);
    for unit in &partitioning {
        let (c, a, b) = (c.at(&unit), a.at(&unit), b.at(&unit));
        let mut sum = c.accumulator::<EA>(
            &a,
            &b,
            microkernel.drained_instruction(),
            Semiring::SUM_PROD,
        );
        sum.zero();
        for step in unit.walk_within(&b) {
            unit_registers_microkernel(&step, &sum, &a, &b);
        }
        sum.drained_into(&c);
    }
}
