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

use cubecl::{CubeDim, Runtime, client::ComputeClient, prelude::*};
use cubek_std::InputBinding;
use cubek_tile::{
    Axis, ByAxis, ComputeScope, Coverage, CubeAxis, Distribution, Partitioner, Space, Spread,
    TileArg, TileArgLaunch,
};

use crate::{
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

    let (a_arg, a_storage) = lhs_layout.tensor_arg(lhs.into_data(), b, m, k);
    let (b_arg, b_storage) = rhs_layout.tensor_arg(rhs.into_data(), b, k, n);
    let (c_arg, c_storage) = out_layout.tensor_arg(out, b, m, n);

    // The full {B, M, N, K} space; each operand carries the batch plus its two
    // matrix axes (batch first, so `partition`'s trailing two axes are the matrix
    // tile and the batch is pinned).
    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]);

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[(B, 1), (M, tile), (N, tile), (K, tile)]),
        ByAxis::new(&[
            (B, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::Z),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (M, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::X),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (N, {
                Distribution::Spatial {
                    scope: ComputeScope::Cube(CubeAxis::Y),
                    spread: Spread::Contiguous,
                    coverage: Coverage::TilesEach(1),
                }
            }),
            (K, Distribution::Sequential),
        ]),
    )
    .staged();

    let space = space.with_partitioner(partitioner.clone());
    let cube_count = partitioner.cube_count(&space);
    let cube_dim = CubeDim::new_single();

    let dtype = dtypes.acc_global;

    mosaic_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        TileArgLaunch::new(a_arg, space.project(&[B, M, K]), a_storage),
        TileArgLaunch::new(b_arg, space.project(&[B, K, N]), b_storage),
        TileArgLaunch::new(c_arg, space.project(&[B, M, N]), c_storage),
        dtype,
    );

    Ok(())
}

#[cube(launch)]
pub fn mosaic_kernel<E: Numeric>(
    a: &TileArg<'_, E>,
    b: &TileArg<'_, E>,
    c: &TileArg<'_, E>,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile();
    let b = b.tile();
    let mut c = c.tile();
    c.mma(&a, &b);
}
