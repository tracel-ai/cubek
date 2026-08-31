//! Launch the quantized decode gemv: build the space the blueprint describes, bind four tensors
//! to it, and run one kernel.
//!
//! Every binding is stated in **values**: the weight's shape and strides count the values its
//! words hold, and `.packed(field)` says how many share one word. Nothing here widens a scale,
//! and nothing carries a quantization scheme — the scales bind at their own element type
//! because they are an ordinary operand.
//!
//! Scalar, though, and that is the orientation rather than a default worth tuning: a lane owns
//! `rows_per_lane` output rows against one block, and those scales sit a row apart in the scales
//! buffer. A wide read wants a lane owning consecutive blocks, which is the opposite of the
//! interleave the fold is built on.

use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek_tile::{
    Buffering, ComputeScope, Coverage, CubeAxis, Cut, Distribution, Instruction, PhysicalAxisMap,
    Projection, Spread, Tiling, WalkOrder,
};

use crate::{
    definition::MatmulSetupError,
    routine::BlueprintStrategy,
    tiled::{
        M, N,
        quant_gemv::{
            base::{QuantGemvProblem, QuantGemvRoutine},
            kernel::quant_gemv_kernel,
            operands::{KB, KI, QuantGemvOperands},
        },
    },
};

/// The element each of the four operands is read at.
///
/// Off the bindings rather than on them, because one of them is not a binding's to answer: the
/// weight's buffer holds `u32` words and `served` is what they decode to, which is also the
/// element the contraction runs in. At `f32` the partials never round-trip through a narrower
/// element between `K` steps, which is what a long walk needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantGemvElems {
    pub served: ElemType,
    pub x: ElemType,
    pub scales: ElemType,
    pub out: ElemType,
}

/// The four tensors a launch binds, each stated in values.
pub struct QuantGemvBindings<R: Runtime> {
    /// The activations, `[rows, d_in]`.
    pub x: TensorBinding<R>,
    /// The packed weight, `[d_out, d_in]` in values over a buffer of `u32` words.
    pub weights: TensorBinding<R>,
    /// The scales, `[d_out, d_in / block]`, at whatever element they are stored in.
    pub scales: TensorBinding<R>,
    /// The result, `[d_out, rows]` — the weight's rows are the output's, this orientation
    /// putting them on the buffer's outer dim.
    pub out: TensorBinding<R>,
}

/// `edge`-wide tiles dealt across `lanes` lanes of the plane.
///
/// Not [`Cut::unit`], twice over: that constructor hardwires [`Spread::Contiguous`] where the
/// fold wants lanes reading neighbouring words, and it defers the lane count to whatever the
/// launcher resolves, where the blueprint derived its edges *from* that count on the host.
fn unit(edge: usize, spread: Spread, lanes: usize) -> Cut {
    Cut::new(
        edge,
        Distribution::Spatial {
            scope: ComputeScope::Unit,
            spread,
            coverage: Coverage::Instances(lanes),
        },
    )
}

/// `y = (W ⊗ s) · x`, one launch.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: QuantGemvBindings<R>,
    problem: &QuantGemvProblem,
    strategy: &BlueprintStrategy<(), QuantGemvRoutine>,
    dtypes: QuantGemvElems,
) -> Result<(), MatmulSetupError> {
    let plane_dim = client.properties().hardware.plane_size_max as usize;
    let blueprint = QuantGemvRoutine::blueprint(strategy, problem, plane_dim)?;

    let QuantGemvBindings {
        x,
        weights,
        scales,
        out,
    } = bindings;
    let (factor, block, blocks) = (problem.factor(), problem.block, problem.blocks());
    let mut ops = QuantGemvOperands::new(dtypes.served, dtypes.x, dtypes.scales, dtypes.out);

    let extents = [
        (M, problem.d_out),
        (N, problem.rows),
        (KB, blocks),
        (KI, block),
    ];
    let space = Tiling::over(&mut ops, &extents)
        // A strip of output rows per cube, walking all of `K`. Nothing is staged: the weight is
        // read exactly once, so there is no reuse for a stage to amortize.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::cube(CubeAxis::X, blueprint.rows_per_cube))
                .axis(N, Cut::sequential(problem.rows))
                .axis(KB, Cut::sequential(blocks))
                .axis(KI, Cut::sequential(block));
        })
        // One plane per group of rows.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::plane(blueprint.rows_per_plane))
                .axis(N, Cut::sequential(problem.rows))
                .axis(KB, Cut::sequential(blocks))
                .axis(KI, Cut::sequential(block));
        })
        // The fold: `rows_per_lane` rows per aligned lane group, the group's lanes interleaving
        // the contraction between them. Each takes one stored word of `KI`, and where a group
        // reaches past one block it takes whole blocks of `KB` — a cut cuts one axis or the
        // other and cannot straddle two. The partials the lanes hold drain inside the plane.
        .instruction(
            Instruction::registers(blueprint.rows_per_lane * factor),
            |l, _| {
                l.axis(
                    M,
                    unit(
                        blueprint.rows_per_lane,
                        Spread::Contiguous,
                        blueprint.groups(),
                    ),
                )
                .axis(N, Cut::sequential(problem.rows))
                .axis(KB, unit(1, Spread::Interleaved, blueprint.block_lanes))
                .axis(
                    KI,
                    unit(factor, Spread::Interleaved, blueprint.inside_lanes),
                );
            },
        )
        .build();

    // Every axis static: the extents fold the address arithmetic to constants, which is what a
    // plan built out of the shape is for.
    let launch = space.launcher_over(client, &[]);

    // `K` is one physical dim that `(KB, KI)` partition, so each operand spanning both says so;
    // the scales span `KB` alone and address it as it stands.
    let split = || PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]);
    let w_op = launch
        .arg(weights)
        .gathered(Projection::new(
            &[M, KB, KI],
            &[PhysicalAxisMap::of(M), split()],
        ))
        .operand(&ops.w)
        .packed(problem.field)
        // The served width, which is a whole word: the buffer binds one word wide.
        .vectorize(factor)
        .build();
    let x_op = launch
        .arg(x)
        .gathered(Projection::new(
            &[N, KB, KI],
            &[PhysicalAxisMap::of(N), split()],
        ))
        .operand(&ops.x)
        // One word's worth of activations a step, so both operands serve the same count and the
        // contraction folds a whole line. Leave it scalar and the engine serves one value a
        // step, unpacks the word's other values and discards them.
        .vectorize(factor)
        .build();
    let s_op = launch.bind(&ops.scales, scales).build();
    // Each lane holds a partial of its group's cell, so the accumulator stays scalar: the fold
    // requires it.
    let out_op = launch.bind(&ops.out, out).build();

    quant_gemv_kernel::launch::<R>(
        client,
        launch.cube_count(),
        launch.cube_dim(),
        x_op.vector_size,
        out_op.vector_size,
        w_op.arg(),
        x_op.arg(),
        s_op.arg(),
        out_op.arg(),
        launch.space().clone(),
        dtypes.served,
        dtypes.x,
        dtypes.scales,
        dtypes.out,
    );
    Ok(())
}
