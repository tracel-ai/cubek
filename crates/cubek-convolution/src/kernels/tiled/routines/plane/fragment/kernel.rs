use cubecl::tile::{
    Partitioning, Region, Scratch, Semiring, StageStorage, Stages, Tile, TileTensor,
};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, plane_fragment_microkernel, plane_fragment_partial,
};
use cubek_std::microkernel::StageRing;

/// Every plane running the plane microkernel on its box of the output, staging a gathered input
/// and an rhs held as `storage`.
#[cube(launch_unchecked)]
pub fn plane_fragment_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] storage: StageStorage,
    #[comptime] ring: StageRing,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile(&partitioning);

    for plane in &partitioning {
        let (a, b, c) = (input.at(&plane), rhs.at(&plane), out.at(&plane));
        let mut sum = c
            .accumulator::<EA>(&a, &b, microkernel.tile_instruction(), Semiring::SUM_PROD)
            .with_scratch(Scratch::None);
        sum.zero();
        plane_fragment_staged_microkernel::<EL>(
            &plane,
            &sum,
            &c,
            &a,
            &b,
            microkernel,
            storage.clone(),
            ring,
        );
        sum.drained_into(&c);
    }
}

/// One plane's tile walked through stages of its own: the plane stages its operands a step of its
/// walk at a time, holding as many stages as `ring` says, and contracts every stage into `sum`
/// with the plane microkernel. `sum`, `c`, `a` and `b` are windowed to the plane; a narrow step's
/// partial is opened over `c`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn plane_fragment_staged_microkernel<EL: Numeric>(
    plane: &Region,
    sum: &Tile,
    c: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] stage: StageStorage,
    #[comptime] ring: StageRing,
) {
    let instruction = microkernel.tile_instruction();
    let step_sum = microkernel.instruction.step_sum();
    let partial = plane_fragment_partial::<EL>(c, a, b, step_sum, instruction);

    let walk = plane.walk();
    let in_flight = ring.stages_in_flight.get();
    let mut stages = Stages::smem(&walk, a, b, stage, in_flight, None);

    stages.walk(walk, ring.prefetch, |slot, step| {
        let sum_step = sum.at(step);
        slot.consume(|a_step, b_step| {
            plane_fragment_microkernel(
                step,
                &sum_step,
                &partial,
                a_step,
                b_step,
                instruction,
                None,
                MatmulOperands::AS_BOUND,
            );
        });
    });
}
