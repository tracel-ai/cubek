//! Launch wiring for the Cmma routine: one entry ([`launch_ref`]) serving both
//! deliveries; the blueprint decides, and only the operand construction differs.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::launch::tma::tma_operand;
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{
    Axis, CubeAxis, Cut, Delivery, Launcher, Leaf, Schedule, Space, Strided, Tiling, Tma,
    TmaTileArgLaunch, WalkOrder,
};

use crate::{
    components::global::read::stride_align_bits,
    definition::{
        AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError, broadcast_batches,
    },
    routines::{
        BlueprintStrategy, DeviceSettings, K, M, N, batch_axis,
        cmma::{
            base::{CmmaBlueprint, CmmaRoutine},
            kernel::cmma_kernel,
        },
    },
};

/// The register accumulate type the routine mandates, independent of how the caller built
/// `dtypes` (some paths build a single-dtype `MatmulElems` that never upgrades): tensor
/// cores accumulate `f16`/`bf16` in `f32`, and the epilogue casts back on drain, so the
/// accumulator is always at least the output's precision. Reuses the canonical upgrade in
/// [`MatmulElems::from_globals`]. Keying both the `MmaConfig` lookup and the kernel's `EA`
/// type on this keeps selection and codegen consistent.
fn mandated_acc(dtypes: &MatmulElems) -> StorageType {
    MatmulElems::from_globals(&dtypes.as_global_elems()).acc_register
}

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

    let blueprint =
        CmmaRoutine::blueprint(strategy, &problem, &device_settings, mandated_acc(dtypes))?;

    // The descriptor requires every non-contiguous stride 16-byte aligned; the problem's
    // strides are synthesized, so check the real bindings here.
    if blueprint.delivery.is_tma() {
        let aligned = |strides: &[usize], dtype: &StorageType| {
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

/// The routine's 4-level space: the cube grid (double-buffered smem stages along `K`);
/// one partition per plane; the contraction-step walk staging each step's operand
/// fragments (`Staged`); the fragment grid the step contracts (`Direct`, walked
/// statically). `batch` lists the surviving (extent > 1) output batch axes, one per
/// cube on `Z`.
fn tile_space(
    blueprint: &CmmaBlueprint,
    (m, n, k): (usize, usize, usize),
    batch: &[(Axis, usize)],
) -> Space {
    let (i, c) = (blueprint.instruction, blueprint.partition);
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;

    let batch_axes: Vec<_> = batch.iter().map(|&(a, _)| a).collect();
    let extents: Vec<_> = batch
        .iter()
        .copied()
        .chain([(M, m), (N, n), (K, k)])
        .collect();

    Tiling::new()
        .extents(&extents)
        .level(WalkOrder::RowMajor, Schedule::DoubleBuffered, |l| {
            l.axes(&batch_axes, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, stage_m))
                .axis(N, Cut::cube(CubeAxis::Y, stage_n))
                .axis(K, Cut::sequential(stage_k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::plane(c.m * i.m))
                .axis(N, Cut::plane(c.n * i.n))
                .axis(K, Cut::sequential(stage_k))
        })
        .level(WalkOrder::RowMajor, Schedule::Staged, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(c.m * i.m))
                .axis(N, Cut::sequential(c.n * i.n))
                .axis(K, Cut::sequential(i.k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batch_axes, Cut::sequential(1))
                .axis(M, Cut::sequential(i.m))
                .axis(N, Cut::sequential(i.n))
                .axis(K, Cut::sequential(i.k))
        })
        .build()
}

/// The one entry for both deliveries: the shared geometry (space, launcher, out arg) is
/// built once, and only the operand construction dispatches on the blueprint's
/// [`Delivery`]. A TMA plan is fully validated by then, so on CUDA it runs or fails to
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
    let (problem, blueprint, out_batches) = setup(client, &lhs, &rhs, &out, strategy, dtypes)?;
    let (m, n, k) = (problem.m, problem.n, problem.k);

    // Output batch dims that survive (extent > 1) ride one-per-cube on Z (none under TMA;
    // the blueprint rejected them).
    let batch: Vec<(Axis, usize)> = (0..out_batches.len())
        .filter(|&p| out_batches[p] > 1)
        .map(|p| (batch_axis(p), out_batches[p]))
        .collect();
    let space = tile_space(&blueprint, (m, n, k), &batch);
    // What every operand of this routine is at the instruction; the plan says only how it is cut.
    let leaf = Leaf::Cmma;

    let launch = space.launcher(client);
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();

    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    let (cube_count, cube_dim) = (launch.cube_count(), launch.cube_dim());

    // The one dispatch Rust forces: pick the compile-time family for the runtime delivery.
    // Either path runs the same kernel body and never branches on the delivery again.
    match blueprint.delivery {
        Delivery::Strided => launch_strided::<R>(
            client,
            &launch,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            &out_batch_axes,
            dtypes,
            leaf,
        ),
        Delivery::Tma => launch_tma::<R>(
            client,
            &launch,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            &out_batch_axes,
            &blueprint,
            dtypes,
            (m, n, k),
            leaf,
        ),
    }

    Ok(())
}

/// The strided path: each operand lined at the widest width the launcher's gate allows,
/// built by the shared [`StridedTileSource`](cubek_tile::StridedTileSource) derivation.
#[allow(clippy::too_many_arguments)]
fn launch_strided<R: Runtime>(
    client: &ComputeClient<R>,
    launch: &Launcher<'_, R>,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    lhs: TensorBinding<R>,
    rhs: TensorBinding<R>,
    out: TensorBinding<R>,
    out_batch_axes: &[Axis],
    dtypes: &MatmulElems,
    leaf: Leaf,
) {
    let operand = |binding: TensorBinding<R>, axes: [Axis; 2], dtype: StorageType| {
        let [outer, inner] = axes;
        let v = launch.vector_size(inner, &[(&binding, &[outer, inner])], dtype.size());
        launch
            .arg(binding)
            .subspace(&[outer, inner])
            .batches(out_batch_axes)
            .vectorize(v)
            .leaf(leaf)
            .build()
    };
    let a = operand(lhs, [M, K], dtypes.lhs_global);
    let b = operand(rhs, [K, N], dtypes.rhs_global);
    let c = operand(out, [M, N], dtypes.acc_global);
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
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
        mandated_acc(dtypes),
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
    lhs: TensorBinding<R>,
    rhs: TensorBinding<R>,
    out: TensorBinding<R>,
    out_batch_axes: &[Axis],
    blueprint: &CmmaBlueprint,
    dtypes: &MatmulElems,
    (m, n, k): (usize, usize, usize),
    leaf: Leaf,
) {
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;
    // A fn, not a closure: each operand instantiates its own erased element type.
    fn operand<E: Numeric, R: Runtime>(
        leaf: Leaf,
        binding: TensorBinding<R>,
        axes: [Axis; 2],
        box_dims: (usize, usize),
        (rows, cols): (u32, u32),
        dtype: StorageType,
    ) -> TmaTileArgLaunch<E, R> {
        let (map, transposed) = tma_operand(
            binding,
            1,
            MatrixLayout::RowMajor,
            box_dims,
            dtype,
            TensorMapSwizzle::None,
        );
        TmaTileArgLaunch::tensor_map(map, &axes, (1, rows, cols), transposed, leaf)
    }
    let a = operand(
        leaf,
        lhs,
        [M, K],
        (stage_m, stage_k),
        (m as u32, k as u32),
        dtypes.lhs_global,
    );
    let b = operand(
        leaf,
        rhs,
        [K, N],
        (stage_k, stage_n),
        (k as u32, n as u32),
        dtypes.rhs_global,
    );
    let v_out = launch.vector_size(N, &[(&out, &[M, N])], dtypes.acc_global.size());
    let c = launch
        .arg(out)
        .subspace(&[M, N])
        .batches(out_batch_axes)
        .vectorize(v_out)
        .leaf(leaf)
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
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
        mandated_acc(dtypes),
    );
}
