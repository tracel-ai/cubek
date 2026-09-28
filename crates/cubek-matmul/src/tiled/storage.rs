//! Storing a matrix in storage tiles, and laying it back out.
//!
//! A storage-tiled tensor is stored one tile at a time, as the [`Layout`] it was stored in
//! states: its tiles, finest first, each made of the one below, then the grid of tiles. Its
//! binding carries how many pieces each dim is stored in (its `tiling`, `[.., R/tr, C/tc, tr, tc]`
//! for one level) and the order they follow one another in (its strides). A routine that reads
//! storage tiles (cmma's stage) moves a tile as one contiguous run.
//!
//! Tiling is a relayout on the tile DSL: a space over the matrix with exactly one level, the
//! outermost storage tile, one cube per tile. Whichever side is storage-tiled is read or written
//! through its layout, however many levels lie inside that tile.

use cubecl::{
    client::Client,
    ir::ElemType,
    prelude::*,
    std::tensor::{TensorHandle, layout::CoordsDyn},
    zspace::{Shape, Strides, Tiling, metadata::Metadata},
};
use cubek_tile::{
    Geometry, Grid, Launcher, Level, Levels, Partitioning, Space, StorageTiling, TileArg,
};

use crate::{
    definition::MatmulSetupError,
    tiled::{batch_axis, logical_dims},
};

/// A matrix's outermost storage tile, `(rows, cols)`: each dim over its grid's count, whatever
/// finer pieces it holds.
type StorageTile = (usize, usize);

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

/// What [`tile`] is told: how the tiles are laid out, stated leaf-up, over axes the caller names.
pub use cubek_tile::{Axis, GridLayout, Layout};

/// The stored matrix's rows and columns as the relayout's own space names them. A caller states
/// its layout in its own axes; only the tile's extents cross over.
const ROWS: Axis = Axis(0);
const COLS: Axis = Axis(1);

/// Store a plain matrix (leading batch dims, trailing two dims that `axes` names) as `layout`
/// states: its tiles, finest first, and the order of the grid of tiles, all in the
/// caller's axes. The result's metadata states the tiling, so its binding says how it is stored
/// and any routine folds its logical shape back; the order is in its strides.
///
/// ```ignore
/// // A [k, n] weight in 16 x 32 tiles, rows of each tile first, the next tile along k.
/// let layout = Layout::tile(&[(N, 32), (K, 16)]).grid(&[K, N]);
/// let stored = tile(&client, weight.binding(), [K, N], dtype, layout)?;
/// ```
///
/// # Errors
///
/// A source that is already storage-tiled (untile it first), a layout that does not close the
/// matrix in whole tiles (a routine reading storage tiles reads whole ones, and a padded buffer
/// would change the logical shape), or one deeper than a tiling records.
#[allow(clippy::result_large_err)]
pub fn tile(
    client: &Client,
    src: TensorBinding,
    axes: [Axis; 2],
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
        .over(&[(axes[0], rows), (axes[1], cols)])
        .map_err(|misfit| refused(misfit.to_string()))?;
    let (matrix, tiling) = layout.physical(&axes);
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
    let fragments: Vec<usize> = batches
        .iter()
        .map(|_| 1)
        .chain([tiling.fragments(0), tiling.fragments(1)])
        .collect();
    let config = |e| refused(format!("{e:?}"));
    let tiling = Tiling::new(&fragments).map_err(config)?;
    let metadata = Metadata::new(Shape::from(shape.clone()), Strides::from(strides))
        .with_tiling(tiling)
        .map_err(config)?;
    let mut dst = TensorHandle::empty(client, Shape::from(shape), dtype);
    dst.metadata = Box::new(metadata);
    let tile = outermost_tile(&dst.clone().binding()).map_err(refused)?;
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
/// A source that is not storage-tiled, or one that stores a batch dim in pieces.
#[allow(clippy::result_large_err)]
pub fn untile(
    client: &Client,
    src: TensorBinding,
    dtype: ElemType,
) -> Result<TensorHandle, MatmulSetupError> {
    let refused = |why: String| MatmulSetupError::InvalidConfig(Box::new(format!("untile: {why}")));
    if !src.tiling.is_tiled() {
        return Err(refused("the source is not storage-tiled".to_string()));
    }
    let tile = outermost_tile(&src).map_err(refused)?;
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

/// The outermost storage tile of a storage-tiled matrix, `(rows, cols)`: each matrix dim's
/// extent over its grid's count. The grid is each dim's coarsest piece, which a tiling lists
/// first, so both sit right after the batch dims however many levels lie under them.
fn outermost_tile(binding: &TensorBinding) -> Result<(usize, usize), String> {
    let rank = binding.shape.len();
    let logical_rank = binding
        .tiling
        .logical_rank(rank)
        .map_err(|e| format!("{e:?}"))?;
    let fragments = binding.tiling.fragments(logical_rank);
    let batches = logical_rank - 2;
    if fragments[..batches].iter().any(|&n| n != 1) {
        return Err(format!(
            "batch dims are stored plain, got {fragments:?} pieces per dim"
        ));
    }
    let (_, rows, cols) = logical_dims(binding);
    Ok((
        rows / binding.shape[batches],
        cols / binding.shape[batches + 1],
    ))
}

/// The labels of a matrix binding's trailing dims: `[ROWS, COLS]`, or one per piece of a
/// storage-tiled one, in the order its tiling lists them.
fn labels_of(binding: &TensorBinding) -> Vec<Axis> {
    match binding.tiling.is_tiled() {
        true => StorageTiling::stored(binding.tiling, 2, binding.shape.len()).order(&[ROWS, COLS]),
        false => vec![ROWS, COLS],
    }
}

/// The one launch both directions share: the matrix's space cut by one level, the outermost
/// storage tile, a cube of one plane per tile; the levels inside it are the tiled side's to
/// decode. Each side's binding says whether it is the tiled one.
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
    // A line runs along the columns of both sides: a tiled side's pieces are labelled as its
    // tiling lists them, so its width is the read it stored, taken whole.
    let (src_labels, dst_labels) = (labels_of(&src), labels_of(&dst));
    let v = match (src_labels.last(), dst_labels.last()) {
        (Some(&COLS), Some(&COLS)) => launch.vector_size(
            COLS,
            &[
                (&Geometry::from(&src), &src_labels),
                (&Geometry::from(&dst), &dst_labels),
            ],
            dtype.size(),
        ),
        _ => 1,
    };
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
