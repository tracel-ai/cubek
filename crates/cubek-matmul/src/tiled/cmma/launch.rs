//! Launch wiring for the Cmma routine: one entry ([`launch_ref`]) serving both
//! deliveries; the blueprint decides, and only the operand construction differs.

use cubecl::{Runtime, client::ComputeClient, ir::FloatKind, prelude::*};
use cubek_std::{
    InputBinding, MatrixLayout,
    launch::tma::{stride_align_bits, tma_operand},
};
use cubek_tile::{
    Axis, Buffering, CubeAxis, Cut, Instruction, Launcher, Operand, Residence, Space, Strided,
    Tiling, Tma, TmaTileArgLaunch, WalkOrder,
};

use crate::{
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulIdent, MatmulProblem, MatmulSetupError,
        broadcast_batches,
    },
    routine::{BlueprintStrategy, DeviceSettings},
    tiled::cmma::{
        base::{CmmaBlueprint, CmmaDelivery, CmmaRoutine},
        kernel::cmma_kernel,
    },
    tiled::{K, M, MatmulOperands, N, batch_axis},
};

/// A cmma operand must be row-major contiguous: the transport addresses each window
/// by a row stride off a scalar offset.
#[allow(clippy::result_large_err)]
fn validate_row_major(strides: &[usize]) -> Result<(), MatmulSetupError> {
    if strides.last() == Some(&1) {
        Ok(())
    } else {
        Err(MatmulSetupError::InvalidConfig(Box::new(
            "Cmma: operand is not row-major contiguous".to_string(),
        )))
    }
}

/// Cmma carries one type per input from global memory down to the fragment (the kernel's
/// `EL`/`ER`), so a stage or register type of its own is a cast this routine does not emit.
/// The accumulator is the exception: its upgrade is plumbed, `EA` plus a cast on drain.
#[allow(clippy::result_large_err)]
fn validate_single_type(dtypes: &MatmulElems, ident: MatmulIdent) -> Result<(), MatmulSetupError> {
    let (global, stage, register) = (
        dtypes.global(ident),
        dtypes.stage(ident),
        dtypes.register(ident),
    );
    if stage == global && register == global {
        Ok(())
    } else {
        Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Cmma runs {ident:?} at {global:?} throughout; stage {stage:?} and register \
             {register:?} would need a cast it does not emit"
        ))))
    }
}

/// The derivation both entries share: reject what the routine can't run, build the
/// [`MatmulProblem`], and resolve the [`CmmaBlueprint`]. Returns the problem, the
/// plan, and the output's broadcast batch shape.
#[allow(clippy::result_large_err, clippy::type_complexity)]
fn setup<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &InputBinding<R>,
    rhs: &InputBinding<R>,
    out: &TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CmmaRoutine>,
    dtypes: &MatmulElems,
    acc: ElemType,
) -> Result<(MatmulProblem, CmmaBlueprint, Vec<usize>), MatmulSetupError> {
    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Cmma does not support quantized inputs".to_string(),
        )));
    }
    validate_row_major(&lhs.data().strides)?;
    validate_row_major(&rhs.data().strides)?;
    validate_row_major(&out.strides)?;

    validate_single_type(dtypes, MatmulIdent::Lhs)?;
    validate_single_type(dtypes, MatmulIdent::Rhs)?;

    // Logical dims off each strided operand: trailing two axes are the matrix, leading
    // dims its own (possibly broadcast) batch shape.
    let split = |shape: &[usize]| {
        let r = shape.len();
        (shape[..r - 2].to_vec(), shape[r - 2], shape[r - 1])
    };
    let (lhs_batches, m, k) = split(lhs.shape());
    let (rhs_batches, _, n) = split(rhs.shape());
    let out_batches = broadcast_batches(&lhs_batches, &rhs_batches).ok_or_else(|| {
        MatmulSetupError::InvalidConfig(Box::new(format!(
            "Cmma: batch shapes do not broadcast, lhs:{lhs_batches:?} rhs:{rhs_batches:?}"
        )))
    })?;

    let sz = dtypes.acc_global.size();
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(sz));

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches[..].into(),
        rhs_batches[..].into(),
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = DeviceSettings {
        client: client.clone(),
        plane_dim: client.properties().hardware.plane_size_max,
        vector_sizes: AvailableVectorSizes::from_type_sizes(client, sz, sz, sz).pick_max()?,
        max_cube_count: client.properties().hardware.max_cube_count,
    };

    let blueprint = CmmaRoutine::blueprint(strategy, &problem, &device_settings, acc)?;

    // The descriptor requires every non-contiguous stride 16-byte aligned; the problem's
    // strides are synthesized, so check the real bindings here.
    if blueprint.delivery.is_tma() {
        let aligned = |strides: &[usize], dtype: &ElemType| {
            stride_align_bits(strides, &MatrixLayout::RowMajor, dtype) >= 4
        };
        if !aligned(&lhs.data().strides, &dtypes.lhs_global)
            || !aligned(&rhs.data().strides, &dtypes.rhs_global)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Cmma TMA: strides must be aligned to 16 bytes".to_string(),
            )));
        }
    }
    Ok((problem, blueprint, out_batches.to_vec()))
}

/// The routine's 4-level space with its operands' residences stated in place: the cube grid
/// (double-buffered along `K`) stages both inputs into shared memory; one partition per
/// plane; the contraction-step walk moves them into cmma fragments; the fragment grid the
/// step contracts is walked statically. The accumulator states its own residence at the cube
/// grid: `out` spans no contracted axis, so that level holds one region per cube, and a
/// register-resident accumulator there spans the whole `K` walk below it. `batch` lists the
/// surviving (extent > 1) output batch axes, one per cube on `Z`.
fn tile_space(
    blueprint: &CmmaBlueprint,
    (m, n, k): (usize, usize, usize),
    batch: &[(Axis, usize)],
    dtypes: &MatmulElems,
    acc: ElemType,
) -> (Space, MatmulOperands) {
    let (i, c) = (blueprint.instruction, blueprint.partition);
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;

    let batch_axes: Vec<_> = batch.iter().map(|&(a, _)| a).collect();
    let extents: Vec<_> = batch
        .iter()
        .copied()
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    let mut ops = MatmulOperands::new(dtypes);
    let space = Tiling::over(&mut ops, &extents)
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l, o| {
            l.axes(&batch_axes, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, stage_m))
                .axis(N, Cut::cube(CubeAxis::Y, stage_n))
                .axis(K, Cut::sequential(stage_k));
            o.a.stage_as(Residence::Smem, dtypes.lhs_stage);
            o.b.stage_as(Residence::Smem, dtypes.rhs_stage);
            o.out.stage_as(Residence::Register, acc);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::plane(c.m * i.m))
                .axis(N, Cut::plane(c.n * i.n))
                .axis(K, Cut::sequential(stage_k));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(c.m * i.m))
                .axis(N, Cut::sequential(c.n * i.n))
                .axis(K, Cut::sequential(i.k));
            o.a.stage_as(Residence::Register, dtypes.lhs_register);
            o.b.stage_as(Residence::Register, dtypes.rhs_register);
        })
        .instruction(Instruction::Cmma, |l, _| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(i.m))
                .axis(N, Cut::sequential(i.n))
                .axis(K, Cut::sequential(i.k));
        })
        .build();
    (space, ops)
}

/// The one entry for both deliveries: the shared geometry (space, launcher, out arg) is
/// built once, and only the operand construction dispatches on the blueprint's
/// [`CmmaDelivery`]. A TMA plan is fully validated by then, so on CUDA it runs or fails to
/// compile, never silently degrades.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), CmmaRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    // The accumulate type the routine mandates, whatever `dtypes` says (some callers build a
    // single-dtype `MatmulElems` that never upgrades): tensor cores accumulate `f16`/`bf16` in
    // `f32`, and the epilogue casts back on drain. Keys the `MmaConfig` lookup, the `out`
    // register stage and the kernel's `EA`, so selection and codegen cannot disagree.
    let acc = match dtypes.acc_global {
        ElemType::Float(FloatKind::F16 | FloatKind::BF16) => f32::elem_type_native(),
        out_elem => out_elem,
    };
    let (problem, blueprint, out_batches) = setup(client, &lhs, &rhs, &out, strategy, dtypes, acc)?;
    let (m, n, k) = (problem.m, problem.n, problem.k);

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z (none under TMA;
    // the blueprint rejected them).
    let batch: Vec<(Axis, usize)> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .map(|p| (batch_axis(p), out_batches[p]))
        .collect();
    let (space, ops) = tile_space(&blueprint, (m, n, k), &batch, dtypes, acc);

    let launch = space.launcher(client);
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();

    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    let (cube_count, cube_dim) = (launch.cube_count(), launch.cube_dim());

    // The one dispatch Rust forces: pick the compile-time family for the runtime delivery.
    // Either path runs the same kernel body and never branches on the delivery again.
    match blueprint.delivery {
        CmmaDelivery::Copy => launch_strided::<R>(
            client,
            &launch,
            cube_count,
            cube_dim,
            &ops,
            lhs,
            rhs,
            out,
            &out_batch_axes,
        ),
        CmmaDelivery::Tma => launch_tma::<R>(
            client,
            &launch,
            cube_count,
            cube_dim,
            &ops,
            lhs,
            rhs,
            out,
            &out_batch_axes,
            &blueprint,
            (m, n, k),
        ),
    }

    Ok(())
}

/// The strided path: each operand lined at the widest width the launcher's gate allows,
/// bound to its [`Operand`](cubek_tile::Operand) by the shared
/// [`StridedTileSource`](cubek_tile::StridedTileSource) derivation.
#[allow(clippy::too_many_arguments)]
fn launch_strided<R: Runtime>(
    client: &ComputeClient<R>,
    launch: &Launcher<'_, R>,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    ops: &MatmulOperands,
    lhs: TensorBinding<R>,
    rhs: TensorBinding<R>,
    out: TensorBinding<R>,
    out_batch_axes: &[Axis],
) {
    let v_a = launch.vector_size(K, &[(&lhs, ops.a.axes())], ops.a.dtype().size());
    let a = launch
        .bind(&ops.a, lhs)
        .batches(out_batch_axes)
        .vectorize(v_a)
        .build();
    let v_b = launch.vector_size(N, &[(&rhs, ops.b.axes())], ops.b.dtype().size());
    let b = launch
        .bind(&ops.b, rhs)
        .batches(out_batch_axes)
        .vectorize(v_b)
        .build();
    let v_c = launch.vector_size(N, &[(&out, ops.out.axes())], ops.out.dtype().size());
    let c = launch
        .bind(&ops.out, out)
        .batches(out_batch_axes)
        .vectorize(v_c)
        .build();
    cmma_kernel::launch::<Strided, R>(
        client,
        cube_count,
        cube_dim,
        a.vector_size,
        b.vector_size,
        c.vector_size,
        a.arg(),
        b.arg(),
        c.arg(),
        launch.space().clone(),
        ops.a.dtype(),
        ops.b.dtype(),
        ops.out.dtype(),
        ops.out.current_dtype(),
    );
}

/// The TMA path: each input rides a tensor map whose box is the stage (scalar; TMA moves
/// whole boxes, so vectorization and the batch-axis list don't apply). The out is strided
/// under either delivery.
#[allow(clippy::too_many_arguments)]
fn launch_tma<R: Runtime>(
    client: &ComputeClient<R>,
    launch: &Launcher<'_, R>,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    ops: &MatmulOperands,
    lhs: TensorBinding<R>,
    rhs: TensorBinding<R>,
    out: TensorBinding<R>,
    out_batch_axes: &[Axis],
    blueprint: &CmmaBlueprint,
    (m, n, k): (usize, usize, usize),
) {
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;
    // A fn, not a closure: each operand instantiates its own erased element type.
    fn operand<E: Numeric, R: Runtime>(
        op: &Operand,
        binding: TensorBinding<R>,
        box_dims: (usize, usize),
        (rows, cols): (u32, u32),
    ) -> TmaTileArgLaunch<E, R> {
        let (map, transposed) = tma_operand(
            binding,
            1,
            MatrixLayout::RowMajor,
            box_dims,
            op.dtype(),
            TensorMapSwizzle::None,
        );
        TmaTileArgLaunch::tensor_map(map, op, (1, rows, cols), transposed)
    }
    let a = operand(&ops.a, lhs, (stage_m, stage_k), (m as u32, k as u32));
    let b = operand(&ops.b, rhs, (stage_k, stage_n), (k as u32, n as u32));
    let v_out = launch.vector_size(N, &[(&out, ops.out.axes())], ops.out.dtype().size());
    let c = launch
        .bind(&ops.out, out)
        .batches(out_batch_axes)
        .vectorize(v_out)
        .build();
    cmma_kernel::launch::<Tma, R>(
        client,
        cube_count,
        cube_dim,
        1,
        1,
        c.vector_size,
        a,
        b,
        c.arg(),
        launch.space().clone(),
        ops.a.dtype(),
        ops.b.dtype(),
        ops.out.dtype(),
        ops.out.current_dtype(),
    );
}
