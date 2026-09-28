//! Storing a matrix in storage tiles, and laying it back out.
//!
//! A storage-tiled tensor is stored one tile at a time, `[.., R/tr, C/tc, tr, tc]`, the tile the
//! innermost two dims; its binding says so through its `tiling`, and the order the tiles follow
//! one another in through its strides, as the [`Layout`](cubek_tile::Layout) it was stored in
//! states. The tile is a level of whatever
//! routine reads it (cmma's stage), so a weight tiled to the plan's stage moves as one
//! contiguous run per stage. Tiling is a relayout on the tile DSL: a space over the matrix with
//! exactly one level, the storage tile, one cube per tile. Whichever side is storage-tiled is
//! read or written as a run; the other through its layout.

use cubecl::{
    client::Client,
    ir::ElemType,
    prelude::*,
    std::tensor::{TensorHandle, layout::CoordsDyn},
    zspace::{Shape, Strides, Tiling, metadata::Metadata},
};
use cubek_tile::{
    Axis, Geometry, Grid, GridLayout, Launcher, Level, Levels, Partitioning, Space, StorageTiling,
    TileArg,
};

use crate::{
    definition::MatmulSetupError,
    tiled::{batch_axis, logical_dims, storage_tile},
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

/// What [`tile`] is told: how the tiles are laid out, stated leaf-up.
pub use cubek_tile::Layout;

/// The stored matrix's rows, one of the two axes a storage [`Layout`] over it names.
pub const ROWS: Axis = Axis(0);
/// The stored matrix's columns, the other.
pub const COLS: Axis = Axis(1);

/// Store a plain matrix (leading batch dims, trailing `rows x cols`) as `layout` states: one
/// storage level over [`ROWS`] and [`COLS`], and the order of the grid of tiles. The result's
/// metadata states the tiling, so its binding says how it is stored and any routine folds its
/// logical shape back; the grid's order is in its strides.
///
/// # Errors
///
/// A source that is already storage-tiled (untile it first), a shape the tile does not divide (a
/// routine reading storage tiles reads whole ones, and a padded buffer would change the logical
/// shape), or a layout of more than one storage level, which the relayout does not write.
#[allow(clippy::result_large_err)]
pub fn tile(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
    layout: GridLayout,
) -> Result<TensorHandle, MatmulSetupError> {
    let refused = |why: String| MatmulSetupError::InvalidConfig(Box::new(format!("tile: {why}")));
    if src.tiling.is_tiled() {
        return Err(refused(
            "the source is already storage-tiled; untile it first".to_string(),
        ));
    }
    let (batches, rows, cols) = logical_dims(&src);
    let layout = layout
        .over(&[(ROWS, rows), (COLS, cols)])
        .map_err(|misfit| refused(misfit.to_string()))?;
    let (matrix, tiling) = layout.physical(&[ROWS, COLS]);
    if tiling != StorageTiling::uniform(2, 1) {
        return Err(refused(format!(
            "{layout:?} is not one storage level over the rows and columns, which is what the \
             relayout writes"
        )));
    }
    let tile = (matrix.shape()[2], matrix.shape()[3]);
    // Batch dims stay plain and coarsest, each a whole run of the matrices finer than it.
    let mut shape: Vec<usize> = batches.clone();
    let mut strides = vec![0; batches.len()];
    let mut run = rows * cols;
    for (b, &extent) in batches.iter().enumerate().rev() {
        strides[b] = run;
        run *= extent;
    }
    shape.extend_from_slice(matrix.shape());
    strides.extend_from_slice(matrix.strides());
    let fragments: Vec<usize> = batches.iter().map(|_| 1).chain([2, 2]).collect();
    let config = |e| refused(format!("{e:?}"));
    let tiling = Tiling::new(&fragments).map_err(config)?;
    let mut dst = TensorHandle::empty(client, Shape::from(shape.clone()), dtype);
    dst.metadata = Box::new(
        Metadata::new(Shape::from(shape), Strides::from(strides))
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
        .chain([(ROWS, rows), (COLS, cols)])
        .collect();
    let space = Space::new(&extents);
    // One level, leaf up: the tile is the leaf and there is a cube for every one of them.
    let level = Levels::leaf(&[(ROWS, tr), (COLS, tc)])
        .cubes(&[ROWS, COLS])
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
        COLS,
        &[
            (&Geometry::from(&src), &[ROWS, COLS]),
            (&Geometry::from(&dst), &[ROWS, COLS]),
        ],
        dtype.size(),
    );
    let s = launch
        .arg(src)
        .axes(&[ROWS, COLS])
        .batches(&all_batch_axes)
        .vectorize(v)
        .build();
    let d = launch
        .arg(dst)
        .axes(&[ROWS, COLS])
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
