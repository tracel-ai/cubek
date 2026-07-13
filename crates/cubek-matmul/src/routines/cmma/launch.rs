//! Launch wiring for the Cmma routine: one entry ([`launch_ref`]) serving both
//! deliveries; the blueprint decides, and only the operand construction differs.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_std::launch::tma::tma_operand;
use cubek_std::{InputBinding, MatrixLayout};
use cubek_tile::{
    Axis, CubeAxis, Cut, Delivery, DeliveryFamily, Launcher, Leaf, Schedule, Space, Strided,
    StridedTileArgLaunch, Tiling, Tma, TmaTileArgLaunch, WalkOrder,
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

    let blueprint = CmmaRoutine::blueprint(strategy, &problem, &device_settings)?;

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
        .leaf(Leaf::Cmma { k: i.k })
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

    let launch = space.launcher(client);
    let lhs = lhs.into_data();
    let rhs = rhs.into_data();

    let out_batch_axes: Vec<Axis> = (0..out_batches.len()).map(batch_axis).collect();
    let (cube_count, cube_dim) = (launch.cube_count(), launch.cube_dim());

    // The one dispatch Rust forces: pick the compile-time family for the runtime delivery.
    // `launch_kernel` runs once for either and never branches on the delivery again.
    match blueprint.delivery {
        Delivery::Strided => launch_kernel::<Strided, R>(
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
        ),
        Delivery::Tma => launch_kernel::<Tma, R>(
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
        ),
    }

    Ok(())
}

/// One operand's launch geometry: the two axes it spans (`outer`, then the innermost
/// contiguous axis TMA boxes and vectorization key on), the TMA box (its stage edges),
/// and the operand's runtime `(rows, cols)`.
struct Operand {
    axes: [Axis; 2],
    box_dims: (usize, usize),
    extent: (u32, u32),
}

/// Host-side counterpart to [`DeliveryFamily`]: how a delivery builds one operand's launch
/// arg. Implemented for the tile crate's family markers. Lives here rather than on
/// `cubek_tile`'s `Delivery` because building the arg reaches `cubek_std`'s [`tma_operand`]
/// and the routine's own axes.
trait OperandLaunch: DeliveryFamily {
    fn operand<E: Numeric, R: Runtime>(
        launch: &Launcher<'_, R>,
        binding: TensorBinding<R>,
        operand: Operand,
        out_batch_axes: &[Axis],
        dtype: StorageType,
    ) -> <Self::Arg<E> as LaunchArg>::RuntimeArg<R>;
}

impl OperandLaunch for Strided {
    /// Line the operand's innermost contiguous axis at the widest width the launcher's gate
    /// allows (the box/extent are TMA-only).
    fn operand<E: Numeric, R: Runtime>(
        launch: &Launcher<'_, R>,
        binding: TensorBinding<R>,
        operand: Operand,
        out_batch_axes: &[Axis],
        dtype: StorageType,
    ) -> StridedTileArgLaunch<'static, E, R> {
        let [outer, inner] = operand.axes;
        let v = launch.vector_size(inner, &[(&binding, &[outer, inner])], dtype.size());
        launch
            .arg(binding)
            .subspace(&[outer, inner])
            .batches(out_batch_axes)
            .vectorize(v)
            .build()
    }
}

impl OperandLaunch for Tma {
    /// Encode a tensor map whose box is the stage; the operand stays scalar (TMA moves
    /// whole boxes, so vectorization and the batch-axis list don't apply).
    fn operand<E: Numeric, R: Runtime>(
        launch: &Launcher<'_, R>,
        binding: TensorBinding<R>,
        operand: Operand,
        _out_batch_axes: &[Axis],
        dtype: StorageType,
    ) -> TmaTileArgLaunch<E, R> {
        let (map, transposed) = tma_operand(
            binding,
            1,
            MatrixLayout::RowMajor,
            operand.box_dims,
            dtype,
            TensorMapSwizzle::None,
        );
        let (rows, cols) = operand.extent;
        TmaTileArgLaunch::tensor_map(
            map,
            launch.space().project(&operand.axes),
            (1, rows, cols),
            transposed,
        )
    }
}

/// The launch body, shared by every delivery: it asks the family `D` for each operand, and
/// the out is strided under either. The args are built here, not returned, so the element
/// types the launch macro erases stay inside one body.
#[allow(clippy::too_many_arguments)]
fn launch_kernel<D: OperandLaunch, R: Runtime>(
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
) {
    let (stage_m, stage_n) = blueprint.stage();
    let stage_k = blueprint.stage_k;
    let a = D::operand(
        launch,
        lhs,
        Operand {
            axes: [M, K],
            box_dims: (stage_m, stage_k),
            extent: (m as u32, k as u32),
        },
        out_batch_axes,
        dtypes.lhs_global,
    );
    let b = D::operand(
        launch,
        rhs,
        Operand {
            axes: [K, N],
            box_dims: (stage_k, stage_n),
            extent: (k as u32, n as u32),
        },
        out_batch_axes,
        dtypes.rhs_global,
    );
    // The out is strided under either delivery: lined at the widest width the launcher's
    // gate allows, labeled with the full output batch-axis list (the builder right-aligns
    // it, numpy broadcast, size-1 dims drop out).
    let v_out = launch.vector_size(N, &[(&out, &[M, N])], dtypes.acc_global.size());
    let c = launch
        .arg(out)
        .subspace(&[M, N])
        .batches(out_batch_axes)
        .vectorize(v_out)
        .build();
    cmma_kernel::launch::<D, R>(
        client,
        cube_count,
        cube_dim,
        a,
        b,
        c,
        dtypes.lhs_global,
        dtypes.rhs_global,
        dtypes.acc_global,
    );
}
