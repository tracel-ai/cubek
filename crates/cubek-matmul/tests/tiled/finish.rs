//! The count of units reaching [`Destination::finish`] is the units each routine launches: a
//! destination that counts them is launched through every routine whose launch reaches it.

use cubecl::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, Type, UIntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::{Bound, Destination, DestinationLaunch, Output};
use cubecl::tile::{Partitioning, Tile};
use cubecl::zspace::shape;
use cubek_matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine,
    CubeStagedTmaFillerMatmulRoutine, CubeStagedTmaMatmulRoutine, CubeStagedWgmmaMatmulRoutine,
    HypercubeLastArrivalMatmulRoutine, MatmulRoutine, MatmulTensors, PlaneShuffleMatmulRoutine,
    UnitRegistersMatmulRoutine, UnitWholeKMatmulRoutine,
    tensor::{PlainTensor, StoredTensor},
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::{Hardware, routine::Routine};
use cubek_test_utils::{StridedLayout, TestInput, TestOutcome, ValidationResult};

/// The buffered output, and the units that reached [`Destination::finish`].
struct Counted;

#[derive(CubeType, CubeLaunch)]
struct CountedArg<'a, E: Numeric> {
    out: Output<'a, E>,
    units: &'a [Atomic<u32>],
}

#[cube]
impl Destination for Counted {
    type Arg<E: Numeric> = CountedArg<'static, E>;

    fn tile<E: Numeric, V: Size>(arg: &Self::Arg<E>, partitioning: &Partitioning) -> Tile {
        arg.out.tile(partitioning)
    }

    fn finish<E: Numeric>(arg: &Self::Arg<E>) {
        arg.units[0].fetch_add(1);
    }
}

struct CountedOperand {
    out: Bound,
    units: BufferArg,
}

impl DestinationLaunch for Counted {
    type Operand = CountedOperand;

    fn arg<E: Numeric>(operand: CountedOperand) -> CountedArgLaunch<'static, E> {
        CountedArgLaunch::new(operand.out.output(), operand.units)
    }
}

/// A product of `[rows, 64] x [64, 32]`: many rows for the routines that tile them, and for the
/// ones that hold at most eight rows whole and sum a float rhs in cols, four against a col rhs.
const MANY_ROWS: (usize, StridedLayout) = (32, StridedLayout::RowMajor);
const FEW_ROWS: (usize, StridedLayout) = (4, StridedLayout::ColMajor);

/// Launch `routine`'s first served setting on `product` through [`Counted`], asserting every unit
/// it launched reached `finish`. A routine declining the product is the error, named.
fn finishes<R, M, SM>(routine: R, (rows, rhs): (usize, StridedLayout)) -> Result<(), String>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
{
    let name = core::any::type_name::<R>()
        .rsplit("::")
        .next()
        .unwrap_or_default();
    let client = cubecl::test_device().client();
    let zeros = |dims, layout| {
        TestInput::builder(client.clone(), dims)
            .dtype(ElemType::Float(FloatKind::F32))
            .layout(layout)
            .zeros()
            .generate_without_host_data()
    };
    let lhs = zeros(shape![rows, 64], StridedLayout::RowMajor);
    let rhs = zeros(shape![64, 32], rhs);
    let out = zeros(shape![rows, 32], StridedLayout::RowMajor);
    let tensors = MatmulTensors::writing_into(
        &client,
        StoredTensor::Plain(PlainTensor::of_handle(lhs)),
        StoredTensor::Plain(PlainTensor::of_handle(rhs)),
        PlainTensor::of_handle(out),
        R::ROWS_KIND,
    )
    .expect("dense float operands bind");
    let hw = Hardware::from_client(&client);
    let (blueprint, grid) = routine
        .prepare_first_served(&hw, &tensors.problem())
        .map_err(|why| format!("{name}: {why}"))?;
    let CubeCount::Static(x, y, z) = grid.launch.cube_count.clone() else {
        panic!("a prepared grid is static");
    };
    let units = x * y * z * grid.launch.units_per_cube();

    let counter = client.create_from_slice(u32::as_bytes(&[0]));
    let binding = tensors
        .bind(&blueprint, &grid)
        .map_out(|out| CountedOperand {
            out,
            // SAFETY: one `u32` the kernel only adds into.
            units: unsafe { BufferArg::from_raw_parts(counter.clone(), 1) },
        });
    routine.dispatch::<Counted, _>(&tensors, binding, blueprint, grid);

    let finished = u32::from_bytes(&client.read_one_unchecked(counter))[0];
    assert_eq!(finished, units, "{name}: units that reached finish");
    Ok(())
}

#[test]
fn every_routine_finishes_every_unit() {
    let client = cubecl::test_device().client();
    let u32_atomic = Type::atomic(ElemType::UInt(UIntKind::U32));
    if !client
        .properties()
        .atomic_type_usage(u32_atomic)
        .contains(AtomicUsage::Add)
    {
        let reason = "device has no u32 atomic add".to_string();
        TestOutcome::Validated(ValidationResult::Skipped(reason)).enforce();
        return;
    }
    let launched = [
        finishes(UnitRegistersMatmulRoutine, MANY_ROWS),
        finishes(UnitWholeKMatmulRoutine, FEW_ROWS),
        finishes(PlaneShuffleMatmulRoutine, FEW_ROWS),
        finishes(CubeShuffleMatmulRoutine, FEW_ROWS),
        finishes(CubeStagedMatmulRoutine, MANY_ROWS),
        finishes(CubeStagedAsyncCopyMatmulRoutine, MANY_ROWS),
        finishes(CubeStagedTmaMatmulRoutine, MANY_ROWS),
        finishes(CubeStagedTmaFillerMatmulRoutine, MANY_ROWS),
        finishes(CubeStagedWgmmaMatmulRoutine, MANY_ROWS),
        finishes(HypercubeLastArrivalMatmulRoutine, MANY_ROWS),
    ];
    assert!(
        launched[0].is_ok(),
        "unit registers serves any product on any device"
    );
    let declined: Vec<_> = launched.into_iter().filter_map(Result::err).collect();
    if !declined.is_empty() {
        let reason = format!("declined: {}", declined.join(", "));
        TestOutcome::Validated(ValidationResult::Skipped(reason)).enforce();
    }
}
