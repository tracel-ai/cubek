use crate::kernels::{
    CubeStagedMatmulRoutine, MatmulBlueprint, MatmulPartitioning, MatmulRoutineProblem,
};
use cubecl::tile::StageStorage;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubek_microkernel::matmul::hypercube::HypercubeStagedMicrokernel;
use cubek_microkernel::matmul::{MatmulOperand, MatmulWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

/// The blueprint of a hypercube routine whose cubes run `microkernel` on `item`, placed as
/// `partitioning` places them and staging as `storage` lays the stages out. Over a float product,
/// each operand as wide as its buffer and the microkernel let it, the output no wider than a line
/// of the sum the cubes meet in. Over a block-split space, the widths a cube staged routine binds
/// there ([`CubeStagedMatmulRoutine::widths_in_blocks`]), each cube contracting the item its
/// microkernel reads.
pub(super) fn hypercube_blueprint(
    hw: &Hardware,
    problem: &MatmulRoutineProblem,
    item: &MatmulWorkItem,
    microkernel: HypercubeStagedMicrokernel,
    partitioning: MatmulPartitioning,
    storage: StageStorage,
) -> MatmulBlueprint<HypercubeStagedMicrokernel, StageStorage> {
    match item.blocks {
        None => {
            assert_eq!(
                partitioning.as_tile().fillers() as usize,
                microkernel.cube.fill.filling_planes,
                "a cube is launched over its launch tiling, beside the planes its fill sets aside"
            );
            let mut vectorization = Vectorization::new();
            microkernel.vectorize(&partitioning.cube_space(), &mut vectorization);
            vectorization.require(VectorRequirement::Access {
                operand: MatmulOperand::Out,
                elements: hw.vector_ceiling(problem.sum()),
            });
            let mut blueprint = MatmulBlueprint::new(
                hw,
                problem,
                microkernel,
                partitioning,
                vectorization,
                storage,
            );
            blueprint.microkernel.cube.fill = blueprint
                .microkernel
                .cube
                .fill
                .at_widths(&blueprint.widths, item);
            blueprint
        }
        Some(_) => MatmulBlueprint {
            widths: CubeStagedMatmulRoutine::widths_in_blocks(
                hw,
                &microkernel.cube.contracted(item),
                &microkernel.cube,
                &partitioning,
                &storage,
            ),
            microkernel,
            partitioning,
            shared_memory: storage,
        },
    }
}
