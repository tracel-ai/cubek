use super::{
    CubeShuffleMatmulQRhsMicrokernel, CubeShuffleMatmulQRhsMicrokernelSettings,
    CubeShuffleQRhsPlane,
};
use crate::microkernel::MatmulPartitioning;
use crate::microkernel::dense::cube::{CubeShuffleMicrokernel, CubeShuffleMicrokernelSettings};
use crate::microkernel::dense::plane::{PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings};
use crate::microkernel::{MatmulOperand, MatmulOperands, MatmulRead, MatmulWorkItem};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, blocked};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Instruction, Levels, Space};

/// The software plane: its units take `K`'s scale blocks in turns, each summing its share in
/// registers, the float product's cube shuffle ([`CubeShuffleMicrokernel`]) over a packed weight.
impl CubeShuffleQRhsPlane for PlaneShuffleMicrokernel {
    const CUBE_DECLINE: &'static str = "the packed cube shuffle needs a plane width the device commits to, a weight whose scales group K, and two turns of K or more for its planes";

    /// The lead, then the counts of planes along `K` measured best, or within five percent of it,
    /// on the Qwen3-4B projections at one, four and eight rows on an RTX PRO 6000, the weight in
    /// rows and in tile order. A cube's planes split `K` for the few long columns a plane alone
    /// would leave the device idle on, and its units take more columns and fewer rows as the rows
    /// grow. Sixteen planes along `K` split the narrow projections of a 4B model at one row best
    /// on an M2 Pro.
    fn cube_settings(
        _hw: &Hardware,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>> {
        use PlaneShuffleMicrokernelSettings as Plane;
        let cube = CubeShuffleMatmulQRhsMicrokernelSettings::new;
        vec![
            cube(Plane::stepping(1, None), 4),
            cube(Plane::stepping(1, None), 8),
            cube(Plane::stepping(2, Some(2)), 2),
            cube(Plane::stepping(2, Some(2)), 4),
            cube(Plane::stepping(4, None), 8),
            cube(Plane::stepping(4, Some(2)), 4),
            cube(Plane::stepping(4, Some(2)), 8),
            cube(Plane::stepping(1, None), 16),
        ]
    }

    fn cube_admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        CubeShuffleMicrokernel::admits(item)
    }

    /// The float product's cube shuffle over the weight's scale blocks
    /// ([`CubeShuffleMicrokernel::candidates`]).
    fn cubes(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeShuffleMatmulQRhsMicrokernelSettings<Self::Settings>,
    ) -> Vec<CubeShuffleMatmulQRhsMicrokernel<Self>> {
        let settings = CubeShuffleMicrokernelSettings {
            plane: settings.plane,
            planes_along_k: settings.planes_along_k,
        };
        CubeShuffleMicrokernel::candidates(hw, item, &settings)
            .map(|cube| CubeShuffleMatmulQRhsMicrokernel {
                plane: cube.plane,
                planes_along_k: cube.planes_along_k,
                meet: cube.meet,
            })
            .collect()
    }

    fn step_levels(&self) -> Levels {
        PlaneShuffleMicrokernel::step_levels(self)
            .expect("candidates hold a plane whose units take K in turns")
    }

    fn column_axis(&self) -> Axis {
        self.column_cut().0
    }

    fn plane_columns(&self) -> usize {
        self.columns()
    }

    fn step_depth(&self, leaf: &Space) -> usize {
        let (blocks, inside) = blocked(K);
        leaf.extent(blocks) * leaf.extent(inside) * self.units_along_k
    }

    fn plane_width(&self) -> usize {
        PlaneShuffleMicrokernel::plane_width(self)
    }

    fn tile_instruction(&self) -> Instruction {
        PlaneShuffleMicrokernel::tile_instruction(self)
    }

    /// As bound: the units sum the weight decoded at `f32`, unrounded.
    fn reads(&self, _: &MatmulWorkItem) -> MatmulOperands<MatmulRead> {
        MatmulOperands::all(MatmulRead::AsBound)
    }

    /// Nothing: against a packed weight every width follows from its packing.
    fn vectorize_in_cube(&self, _: &MatmulPartitioning, _: &mut Vectorization<MatmulOperand>) {}

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
