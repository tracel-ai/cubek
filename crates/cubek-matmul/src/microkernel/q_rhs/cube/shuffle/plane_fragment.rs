use super::{
    CubeShuffleMatmulQRhsMicrokernel, CubeShuffleMatmulQRhsMicrokernelSettings,
    CubeShuffleQRhsPlane,
};
use crate::microkernel::dense::cube::CubeMeet;
use crate::microkernel::dense::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use crate::microkernel::{
    MatmulOperand, MatmulOperands, MatmulPartitioning, MatmulRead, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Instruction, Levels, Space};

/// Steps of `K` the tensor-core plane's window holds: eight of the decoding instruction's, so
/// each unit's run of it is thirty-two values, sixteen bytes of codes, and a tile of tile order's
/// depth.
const WINDOW_STEPS: usize = 8;

/// The tensor-core plane: it contracts fragments over windows of `K`, which the cube's planes
/// split, decoding the weight as each fragment loads.
impl CubeShuffleQRhsPlane for PlaneFragmentMicrokernel {
    const CUBE_DECLINE: &'static str = "the packed cube shuffle's plane fragment needs the device's 16-bit 16x8x16 manual mma on a 32-unit plane, columns whole fragments of eight, and K whole windows each unit reads in whole loads";

    /// Four, eight and sixteen planes, which split the Qwen3-4B projections best from one to
    /// sixteen rows on the RTX PRO 6000: one fragment a plane nearly everywhere, two at eight
    /// planes beside it. None on a device that does not offer the plane's decoding instruction:
    /// there they would only be tried and declined.
    fn cube_settings(
        hw: &Hardware,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>> {
        let cube = CubeShuffleMatmulQRhsMicrokernelSettings::new;
        let fragments = |along_n| PlaneFragmentMicrokernelSettings::windowed(along_n, WINDOW_STEPS);
        match Self::decodes_on(hw) {
            true => vec![
                cube(fragments(1), 8),
                cube(fragments(2), 8),
                cube(fragments(1), 4),
                cube(fragments(2), 4),
                cube(fragments(1), 16),
            ],
            false => Vec::new(),
        }
    }

    fn cube_admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        Self::admits(item)
    }

    /// The plane's candidates, its admission included, the settings' planes capped at the
    /// windows of `K` it takes.
    fn cubes(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernel<Self>> {
        if Self::admits(item).is_err() {
            return Vec::new();
        }
        let meet = CubeMeet::of(hw);
        let target = settings.planes_along_k.get();
        Self::candidates(hw, item, &settings.plane)
            .filter_map(|plane| {
                let windows = item.blocks?.k / plane.tile_extent(plane.axes.k);
                Some(CubeShuffleMatmulQRhsMicrokernel {
                    plane,
                    planes_along_k: NonZeroUsize::new(target.min(windows))?,
                    meet,
                })
            })
            .collect()
    }

    fn step_levels(&self) -> Levels {
        PlaneFragmentMicrokernel::step_levels(self)
            .expect("candidates hold a plane against a packed weight")
    }

    fn column_axis(&self) -> Axis {
        self.axes.n
    }

    fn plane_columns(&self) -> usize {
        self.tile_extent(self.axes.n)
    }

    fn step_depth(&self, _: &Space) -> usize {
        self.tile_extent(self.axes.k)
    }

    /// The width its decoding instruction's lane runs are laid over, the only one its candidates
    /// run on.
    fn plane_width(&self) -> usize {
        Self::DECODING_PLANE_UNITS
    }

    fn tile_instruction(&self) -> Instruction {
        PlaneFragmentMicrokernel::tile_instruction(self)
    }

    /// Every packed input dequantized at the element they decode into, a fragment's two operands
    /// being one element ([`MatmulWorkItem::decoding_reads`]).
    fn reads(&self, item: &MatmulWorkItem) -> MatmulOperands<MatmulRead> {
        item.decoding_reads()
    }

    fn vectorize_in_cube(
        &self,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        self.vectorize(&(), vectorization);
    }

    fn estimate_in_cube(
        &self,
        item: &MatmulWorkItem,
        _: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.estimate(item, &(), planes, estimate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::devices;

    /// The tensor-core plane's cube settings are listed only where the device offers its decoding
    /// instruction: elsewhere they would only be tried and declined.
    #[test]
    fn tensor_core_planes_only_where_the_instruction_is_offered() {
        for hw in [
            devices::strix_halo(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::cpu_12(),
        ] {
            assert_eq!(
                !PlaneFragmentMicrokernel::cube_settings(&hw).is_empty(),
                PlaneFragmentMicrokernel::decodes_on(&hw),
                "{hw:?}"
            );
        }
    }
}
