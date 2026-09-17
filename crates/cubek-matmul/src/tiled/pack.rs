//! Packing a matrix into storage tiles, and back.
//!
//! A storage-tiled tensor is stored one tile at a time, `[.., R/tr, C/tc, tr, tc]`, the tile the
//! innermost two dims; its binding says so through its `tiling`. The tile is a level of whatever
//! routine reads it (cmma's stage), so a weight packed to the plan's stage moves as one
//! contiguous run per stage. Packing is a relayout on the tile DSL: a space over the matrix with
//! exactly one level, the storage tile, one cube per tile. Whichever side is storage-tiled is
//! read or written as a run; the other through its layout.

use cubecl::{
    client::Client,
    ir::ElemType,
    prelude::*,
    std::tensor::{TensorHandle, layout::CoordsDyn},
    zspace::{Shape, Tiling},
};
use cubek_tile::{
    Axis, Geometry, KernelForm, Launcher, Level, Partitioning, Space, TileArg, Tiling as Levels,
};

use crate::{
    definition::MatmulSetupError,
    tiled::{M, N, batch_axis, logical_dims, storage_tile},
};

/// The storage tile `(rows, cols)` a matrix is packed into: what its innermost two physical
/// dims hold.
pub type StorageTile = (usize, usize);

/// `dst = src` over their logical cells, one cube per region of `level`, the cube's units
/// striding the region's lines. A side inside its storage tile is a run; the other walks its
/// layout.
#[cube(launch)]
fn relayout<E: Numeric, V: Size>(
    src: &TileArg<'_, E, V>,
    dst: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(comptime!(space.clone()));
    let dst = dst.tile(comptime!(space.clone()));
    let rank = comptime!(space.rank());
    let ri = comptime!(rank - 2);
    let ci = comptime!(rank - 1);
    for cube in space.over(&level) {
        let s = src.at(&cube);
        let mut d = dst.at(&cube);
        let r = s.view::<V>();
        let mut w = d.view_mut::<V>();
        let shape = r.shape();
        let rows = shape[ri];
        let cols = shape[ci];
        let total = rows * cols;
        let mut i = UNIT_POS;
        while i < total {
            // The cube's region is one cell of every batch axis.
            let mut pos = CoordsDyn::new();
            #[unroll]
            for _ in 0..ri {
                pos.push(0u32);
            }
            pos.push(i / cols);
            pos.push(i % cols);
            w.write(pos.clone(), r.read(pos));
            i += CUBE_DIM;
        }
    }
}

/// Pack a plain matrix (leading batch dims, trailing `rows x cols`) into `tile` storage tiles.
/// The result's metadata states the tiling, so its binding says how it is stored and any
/// routine folds its logical shape back.
///
/// # Errors
///
/// A source that is already storage-tiled (unpack it first), or a shape `tile` does not divide:
/// a routine reading storage tiles reads whole ones, and a padded buffer would change the
/// logical shape.
#[allow(clippy::result_large_err)]
pub fn pack(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
    tile: StorageTile,
) -> Result<TensorHandle, MatmulSetupError> {
    if src.tiling.is_tiled() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pack: the source is already storage-tiled; unpack it first".to_string(),
        )));
    }
    let (batches, rows, cols) = logical_dims(&src);
    let (tr, tc) = tile;
    if !rows.is_multiple_of(tr) || !cols.is_multiple_of(tc) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "pack: a {rows}x{cols} matrix is not whole {tr}x{tc} storage tiles; a routine reads \
             whole tiles, so state a tile that divides it"
        ))));
    }
    let physical: Vec<usize> = batches
        .iter()
        .copied()
        .chain([rows / tr, cols / tc, tr, tc])
        .collect();
    let mut dst = TensorHandle::empty(client, Shape::from(physical), dtype);
    dst.metadata = Box::new(
        dst.metadata
            .as_ref()
            .clone()
            .with_tiling(packed_tiling(&batches)?)
            .map_err(|e| MatmulSetupError::InvalidConfig(Box::new(format!("pack: {e:?}"))))?,
    );
    pack_into(client, src, dst.clone().binding(), dtype)?;
    Ok(dst)
}

/// The tiling a packed matrix carries: its batch dims plain, both matrix dims one nesting deep.
#[allow(clippy::result_large_err)]
fn packed_tiling(batches: &[usize]) -> Result<Tiling, MatmulSetupError> {
    let fragments: Vec<usize> = batches.iter().map(|_| 1).chain([2, 2]).collect();
    Tiling::new(&fragments)
        .map_err(|e| MatmulSetupError::InvalidConfig(Box::new(format!("pack: {e:?}"))))
}

/// [`pack`] into a destination the caller allocated: the same relayout, writing where it says.
///
/// What a caller assembling several regions into one allocation needs — a quantized weight is
/// one buffer holding its values and, at an offset, its scales, and each is packed on its own
/// — and it is what [`pack`] does once the allocation is out of the way.
///
/// `dst` states the tile: it is storage-tiled, and its two innermost dims are the tile the
/// source is cut into.
///
/// # Errors
///
/// A source already storage-tiled, a destination that is not, or a pair whose logical shapes
/// disagree — the relayout moves cells between two views of one matrix, so a destination
/// standing for a different one is a caller's mistake rather than a shape to pad.
#[allow(clippy::result_large_err)]
pub fn pack_into(
    client: &Client,
    src: TensorBinding,
    dst: TensorBinding,
    dtype: ElemType,
) -> Result<(), MatmulSetupError> {
    if src.tiling.is_tiled() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pack: the source is already storage-tiled; unpack it first".to_string(),
        )));
    }
    let Some(tile) = storage_tile(&dst, "pack")? else {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pack: the destination is not storage-tiled, so it states no tile to pack into"
                .to_string(),
        )));
    };
    let (batches, rows, cols) = logical_dims(&src);
    let (dst_batches, dst_rows, dst_cols) = logical_dims(&dst);
    if (&batches, rows, cols) != (&dst_batches, dst_rows, dst_cols) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "pack: the destination stands for a {dst_batches:?} {dst_rows}x{dst_cols} matrix \
             and the source for a {batches:?} {rows}x{cols} one"
        ))));
    }
    let (tr, tc) = tile;
    if !rows.is_multiple_of(tr) || !cols.is_multiple_of(tc) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "pack: a {rows}x{cols} matrix is not whole {tr}x{tc} storage tiles; a routine reads \
             whole tiles, so state a tile that divides it"
        ))));
    }
    relayout_launch(client, src, dst, dtype, &batches, (rows, cols), tile);
    Ok(())
}

/// Unpack a storage-tiled matrix back to a plain row-major one.
///
/// # Errors
///
/// A source that is not storage-tiled, or tiled in a way [`storage_tile`] cannot read.
#[allow(clippy::result_large_err)]
pub fn unpack(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
) -> Result<TensorHandle, MatmulSetupError> {
    let Some(tile) = storage_tile(&src, "unpack")? else {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "unpack: the source is not storage-tiled".to_string(),
        )));
    };
    let (batches, rows, cols) = logical_dims(&src);
    let shape: Vec<usize> = batches.iter().copied().chain([rows, cols]).collect();
    let dst = TensorHandle::empty(client, Shape::from(shape), dtype);
    relayout_launch(
        client,
        src,
        dst.clone().binding(),
        dtype,
        &batches,
        (rows, cols),
        tile,
    );
    Ok(dst)
}

/// The one launch both directions share: the matrix's space cut by one level, the storage
/// tile, a cube of one plane per tile. Each side's binding says whether it is the tiled one.
fn relayout_launch(
    client: &Client,
    src: TensorBinding,
    dst: TensorBinding,
    dtype: ElemType,
    batches: &[usize],
    (rows, cols): (usize, usize),
    (tr, tc): StorageTile,
) {
    let batch: Vec<(Axis, usize)> = batches
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b > 1)
        .map(|(i, &b)| (batch_axis(i), b))
        .collect();
    let batch_axes: Vec<Axis> = batch.iter().map(|&(a, _)| a).collect();
    let all_batch_axes: Vec<Axis> = (0..batches.len()).map(batch_axis).collect();
    let extents: Vec<(Axis, usize)> = batch
        .iter()
        .copied()
        .chain([(M, rows), (N, cols)])
        .collect();
    let space = Space::new(&extents);
    // One level, leaf up: the tile is the leaf and there is a cube for every one of them.
    let level = Levels::leaf(&[(M, tr), (N, tc)])
        .cubes(&[M, N])
        .batches(&batch_axes)
        .levels()
        .remove(0);
    let partitioning = Partitioning::new(space, vec![level.clone()]);
    let cube_count = partitioning.cube_count();
    let cube_dim = CubeDim::new_1d(client.properties().hardware.plane_size_max);
    let launch = Launcher::partitioned(
        client,
        partitioning,
        (cube_count.clone(), cube_dim),
        KernelForm::Dynamic,
    );
    let v = launch.vector_size(
        N,
        &[
            (&Geometry::from(&src), &[M, N]),
            (&Geometry::from(&dst), &[M, N]),
        ],
        dtype.size(),
    );
    let s = launch
        .arg(src)
        .subspace(&[M, N])
        .batches(&all_batch_axes)
        .vectorize(v)
        .build();
    let d = launch
        .arg(dst)
        .subspace(&[M, N])
        .batches(&all_batch_axes)
        .vectorize(v)
        .build();
    relayout::launch(
        client,
        cube_count,
        cube_dim,
        v,
        s.arg(),
        d.arg(),
        launch.partitioning_arg(),
        level,
        dtype,
    );
}
