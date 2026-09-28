//! Storing a matrix in storage tiles, and laying it back out.
//!
//! A storage-tiled tensor is stored one tile at a time, `[.., R/tr, C/tc, tr, tc]`, the tile the
//! innermost two dims; its binding says so through its `tiling`. The tile is a level of whatever
//! routine reads it (cmma's stage), so a weight tiled to the plan's stage moves as one
//! contiguous run per stage. Tiling is a relayout on the tile DSL: a space over the matrix with
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
    Axis, Geometry, Launcher, Level, Levels, Partitioning, Space, TileArg, launch::Grid,
};

use crate::{
    definition::MatmulSetupError,
    tiled::{M, N, batch_axis, logical_dims, storage_tile},
};

/// The storage tile `(rows, cols)` a matrix is stored in: what its innermost two physical
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
    let rank = comptime!(space.space().rank());
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

/// Store a plain matrix (leading batch dims, trailing `rows x cols`) in `tile` storage tiles.
/// The result's metadata states the tiling, so its binding says how it is stored and any
/// routine folds its logical shape back.
///
/// # Errors
///
/// A source that is already storage-tiled (untile it first), or a shape `tile` does not divide:
/// a routine reading storage tiles reads whole ones, and a padded buffer would change the
/// logical shape.
#[allow(clippy::result_large_err)]
pub fn tile(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
    tile: StorageTile,
) -> Result<TensorHandle, MatmulSetupError> {
    if src.tiling.is_tiled() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "tile: the source is already storage-tiled; untile it first".to_string(),
        )));
    }
    let (batches, rows, cols) = logical_dims(&src);
    let (tr, tc) = tile;
    if !rows.is_multiple_of(tr) || !cols.is_multiple_of(tc) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "tile: a {rows}x{cols} matrix is not whole {tr}x{tc} storage tiles; a routine reads \
             whole tiles, so state a tile that divides it"
        ))));
    }
    let physical: Vec<usize> = batches
        .iter()
        .copied()
        .chain([rows / tr, cols / tc, tr, tc])
        .collect();
    // The tiling the result carries: its batch dims plain, both matrix dims one nesting deep.
    let fragments: Vec<usize> = batches.iter().map(|_| 1).chain([2, 2]).collect();
    let config = |e| MatmulSetupError::InvalidConfig(Box::new(format!("tile: {e:?}")));
    let tiling = Tiling::new(&fragments).map_err(config)?;
    let mut dst = TensorHandle::empty(client, Shape::from(physical), dtype);
    dst.metadata = Box::new(
        dst.metadata
            .as_ref()
            .clone()
            .with_tiling(tiling)
            .map_err(config)?,
    );
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

/// Lay a storage-tiled matrix back out as a plain row-major one.
///
/// # Errors
///
/// A source that is not storage-tiled, or tiled in a way [`storage_tile`] cannot read.
#[allow(clippy::result_large_err)]
pub fn untile(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
) -> Result<TensorHandle, MatmulSetupError> {
    let Some(tile) = storage_tile(&src, "untile")? else {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "untile: the source is not storage-tiled".to_string(),
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
        .build()
        .remove(0);
    let partitioning = Partitioning::new(space, vec![level.clone()]);
    let cube_count = partitioning.cube_count();
    let cube_dim = CubeDim::new_1d(client.properties().hardware.plane_size_max);
    let launch = {
        let partitioning = partitioning;
        let concrete = partitioning.space().clone();
        let (cube_count, cube_dim) = (cube_count.clone(), cube_dim);
        Launcher::new(
            client,
            partitioning.all_dynamic(),
            &concrete,
            Grid::Stated {
                cube_count,
                cube_dim,
            },
        )
    };
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
        .axes(&[M, N])
        .batches(&all_batch_axes)
        .vectorize(v)
        .build();
    let d = launch
        .arg(dst)
        .axes(&[M, N])
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
