//! Launch wiring for the Mosaic matmul.
//!
//! Mosaic does not go through `MatmulArgs`/`Routine`: it builds three tile-DSL
//! [`Tile`]s straight from the input bindings and launches a kernel whose whole
//! body is `c.mma(a, b)`. It builds the [`MatmulProblem`] from the bindings,
//! reads each operand's [`InnerLayout`] off its strides, and lets the layout
//! build the matching view — so the same launch handles row-, col-, or
//! batch-major inputs. The partitioner cuts the matrix axes into
//! `tile_size`-square sub-tiles (one output tile per cube, M→X / N→Y) and rides
//! the batch on cube Z.

use cubecl::{CubeDim, Runtime, client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;
use cubek_tile::{
    Axis, ByAxis, ComputePrimitive, Coverage, CubeDimension, Distribution, Partitioner, Space,
    Spread, TileKind, TileLaunch, cube_count_for,
};

use crate::{
    components::batch::mosaic::mosaic_kernel,
    definition::{InnerLayout, MatmulElems, MatmulProblem, MatmulSetupError},
    routines::mosaic::MosaicStrategy,
};

// Matmul's axes — the labels this client gives the engine's opaque `Axis`. `B`
// is the (leading) batch axis; `M`/`N`/`K` are the matrix axes, with `K`
// contracted.
const B: Axis = Axis(0);
const M: Axis = Axis(1);
const N: Axis = Axis(2);
const K: Axis = Axis(3);

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &MosaicStrategy,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    if matches!(lhs, InputBinding::Quantized { .. })
        || matches!(rhs, InputBinding::Quantized { .. })
    {
        return Err({
            let msg = "Mosaic does not support quantized inputs".to_string();
            MatmulSetupError::InvalidConfig(Box::new(msg))
        });
    }

    // One element type drives the whole kernel, so the operands must agree.
    if dtypes.lhs_global != dtypes.rhs_global || dtypes.lhs_global != dtypes.acc_global {
        return Err({
            let msg = format!(
                "Mosaic requires a single dtype, got lhs:{:?} rhs:{:?} acc:{:?}",
                dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global
            );
            MatmulSetupError::InvalidConfig(Box::new(msg))
        });
    }

    let lhs_shape = lhs.shape().clone();
    let rhs_shape = rhs.shape().clone();
    let out_shape = out.shape.clone();
    let lhs_strides = lhs.data().strides.clone();
    let rhs_strides = rhs.data().strides.clone();
    let out_strides = out.strides.clone();

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.clone(),
        rhs_shape.clone(),
        out_shape.clone(),
        lhs_strides.clone(),
        rhs_strides.clone(),
        out_strides.clone(),
        dtypes.as_global_elems(),
        address_type,
        lhs.scheme(),
        rhs.scheme(),
    )?;
    let (m, n, k, b) = (problem.m, problem.n, problem.k, problem.num_batches());

    let tile = strategy.tile_size;
    if !m.is_multiple_of(tile) || !n.is_multiple_of(tile) || !k.is_multiple_of(tile) {
        return Err({
            let msg = format!("Mosaic requires tile_size={tile} to divide M={m}, N={n}, K={k}");
            MatmulSetupError::InvalidConfig(Box::new(msg))
        });
    }

    // The inner layout of each operand, read off its strides.
    let lhs_layout = InnerLayout::from_shape_and_strides(&lhs_shape, &lhs_strides);
    let rhs_layout = InnerLayout::from_shape_and_strides(&rhs_shape, &rhs_strides);
    let out_layout = InnerLayout::from_shape_and_strides(&out_shape, &out_strides);

    let a_view = lhs_layout.view(lhs.into_data(), b, m, k);
    let b_view = rhs_layout.view(rhs.into_data(), b, k, n);
    let c_view = out_layout.view(out, b, m, n);

    // The full {B, M, N, K} space; each operand carries the batch plus its two
    // matrix axes (batch first, so `partition`'s trailing-two leaf is the matrix
    // tile and the batch is pinned).
    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]);

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(B, 1), (M, tile), (N, tile), (K, tile)]),
        ByAxis::new(&[
            (B, {
                let dim = CubeDimension::Z;
                Distribution::Spatial {
                    unit: ComputePrimitive::Cube(dim),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (M, {
                let dim = CubeDimension::X;
                Distribution::Spatial {
                    unit: ComputePrimitive::Cube(dim),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (N, {
                let dim = CubeDimension::Y;
                Distribution::Spatial {
                    unit: ComputePrimitive::Cube(dim),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (K, Distribution::Sequential),
        ]),
    );

    let cube_count = cube_count_for(&partitioner, &space);
    let cube_dim = CubeDim::new_single();

    let dtype = dtypes.acc_global;
    let vector_size = 1;

    mosaic_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        TileLaunch::new(
            a_view,
            partitioner.launch(),
            space.select(&[B, M, K]),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            b_view,
            partitioner.launch(),
            space.select(&[B, K, N]),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            c_view,
            partitioner.launch(),
            space.select(&[B, M, N]),
            TileKind::GmemWhole,
        ),
        dtype,
        vector_size,
    );

    Ok(())
}
