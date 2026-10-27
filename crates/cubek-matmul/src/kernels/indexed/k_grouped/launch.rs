//! The K-grouped product's launch, `out[g] = lhs[rows of g]ᵀ @ rhs[rows of g]`: the dense cube
//! staged routine at one setting, each group a batch summing its run of rows
//! ([`KGroupedMatmulTensors`]). A device it declines, as one offering no fragment instruction at
//! the element, is the caller's to run another way.

use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::CubeOrder;
use cubek_microkernel::matmul::dense::cube::{CubeStagedFill, CubeStagedMicrokernelSettings};
use cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernelSettings;
use cubek_microkernel::matmul::staged_walk::MatmulPlaneLayout;
use cubek_microkernel::staging::StageRing;
use cubek_std::Hardware;

use crate::kernels::indexed::k_grouped::KGroupedMatmulTensors;
use crate::kernels::{CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings};

/// The setting a K-grouped product runs at: the dense default's box, one instruction deep, so a
/// run that starts or ends mid-stage reads at most one instruction's rows of zeros at each end;
/// its cubes in the grid's order, its planes each on a tile of their own, and its stages filled
/// by the cube's units, the one walk a group's run of stages is summed through.
pub const K_GROUPED_SETTINGS: CubeStagedMatmulRoutineSettings = CubeStagedMatmulRoutineSettings {
    microkernel: CubeStagedMicrokernelSettings {
        plane: PlaneFragmentMicrokernelSettings::DEFAULT,
        planes: MatmulPlaneLayout::new(2, 4),
        ring: StageRing::SINGLE,
        fill: CubeStagedFill::BY_UNITS,
        ..CubeStagedMicrokernelSettings::DEFAULT
    },
    cube_order: CubeOrder::RowMajor,
};

/// The K-grouped product `tensors` states, laid out `[m, groups · n]` with group `g` owning
/// columns `g · n..(g + 1) · n`, as the stack an M-grouped product reads is: the stack's gradient
/// where the lhs is that product's input and the rhs its output's gradient. A group owning no row
/// gets a zero slab.
///
/// # Errors
///
/// Why the device cannot run the product at [`K_GROUPED_SETTINGS`].
pub fn launch(tensors: &KGroupedMatmulTensors) -> Result<TensorHandle, Refusal> {
    let hw = Hardware::from_client(tensors.client());
    tensors.launch(&hw, &CubeStagedMatmulRoutine, &K_GROUPED_SETTINGS)
}
