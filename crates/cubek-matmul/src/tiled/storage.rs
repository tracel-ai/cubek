//! Storing a tensor in storage tiles, and laying it back out.
//!
//! A storage-tiled tensor is stored as the [`StoragePartitioning`] it was stored in states: its
//! tiles, finest first, then the grid of tiles. Its binding carries how many pieces each dim is
//! stored in (its `tiling`) and the order they follow one another in (its strides).
//!
//! Tiling is a relayout on the tile DSL: a space over the named axes with one level, the
//! outermost stored tile, one cube per tile. Whichever side is storage-tiled is read or written
//! through its layout, however many levels lie inside that tile.

use cubecl::{
    client::Client,
    ir::ElemType,
    prelude::*,
    std::tensor::{TensorHandle, layout::CoordsDyn},
    zspace::{Shape, Strides, Tiling, metadata::Metadata},
};
use cubek_tile::{
    Geometry, Launcher, Level, Levels, Partitioning, Space, TileArg, kind::Field, launch::Grid,
    layout::StorageMisfit,
};

use crate::definition::MatmulSetupError;

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
    for cube in space.over(&level) {
        let s = src.at(&cube);
        let mut d = dst.at(&cube);
        let r = s.view::<V>();
        let mut w = d.view_mut::<V>();
        let shape = r.shape();
        let mut total = 1u32;
        #[unroll]
        for p in 0..rank {
            total *= shape[p];
        }
        let mut i = UNIT_POS;
        while i < total {
            // The cell's coordinates, the last axis fastest.
            let mut pos = CoordsDyn::new();
            #[unroll]
            for p in 0..rank {
                let mut finer = 1u32;
                #[unroll]
                for q in p + 1..rank {
                    finer *= shape[q];
                }
                pos.push((i / finer) % shape[p]);
            }
            w.write(pos.clone(), r.read(pos));
            i += CUBE_DIM;
        }
    }
}

/// What [`tile`] is told: how the matrix is laid down in memory, stated leaf-up over axes the
/// caller names.
pub use cubek_tile::{
    Axis,
    layout::{StorageLevels, StoragePartitioning},
};

/// Store a tensor as `storage` states, its trailing dims named by `axes` (the dims ahead stay
/// plain and coarsest, one batch each). `packed` says the values sit `field.per_word()` to a `u32`
/// along the last of `axes`: the word must then be the finest stored tile, and the words move
/// whole. `src` and the result are stated in values, as every packed binding is.
///
/// ```ignore
/// // NVFP4 values [n, k/16, 16], 8 to a word along KI, loaded 16 values by 2 columns.
/// let storage = StorageLevels::new(&[(KI, 8)]).tile(&[(KI, 2), (N, 2)]).grid(&[KI, KB, N]);
/// let stored = tile(&client, w.binding(), &[N, KB, KI], u32, Some(E2M1.into()), storage)?;
/// ```
///
/// # Errors
///
/// A source already storage-tiled, a packed source whose word is not the finest stored tile or
/// whose last named dim is not its contiguous one, or a partitioning that does not close the
/// tensor in whole tiles or is deeper than a tiling records.
#[allow(clippy::result_large_err)]
pub fn tile(
    client: &Client,
    src: TensorBinding,
    axes: &[Axis],
    dtype: ElemType,
    packed: Option<Field>,
    storage: StoragePartitioning,
) -> Result<TensorHandle, MatmulSetupError> {
    let refused = |why: String| MatmulSetupError::InvalidConfig(Box::new(format!("tile: {why}")));
    if src.tiling.is_tiled() {
        return Err(refused(
            "the source is already storage-tiled; untile it first".to_string(),
        ));
    }
    let rank = src.shape.len();
    let batches = rank - axes.len();
    let values: Vec<(Axis, usize)> = axes
        .iter()
        .copied()
        .zip(src.shape[batches..].iter().copied())
        .collect();
    // A packed tensor is relaid out as its words: the same statement with the word dropped, the
    // packed axis counted in words.
    let (per_word, words_storage) = match packed {
        None => (1, storage.clone()),
        Some(field) => {
            let (packed_axis, per_word) = (axes[axes.len() - 1], field.per_word());
            if storage.tiles().first() != Some(&(packed_axis, per_word)) {
                return Err(refused(format!(
                    "a {field:?} word is {per_word} values along {packed_axis:?}, so it is the \
                     finest stored tile; the statement starts with {:?}",
                    storage.tiles().first()
                )));
            }
            if src.strides[rank - 1] != 1 {
                return Err(refused(format!(
                    "the words pack along {packed_axis:?}, which is not the source's contiguous dim"
                )));
            }
            let words = StorageLevels::new(&storage.tiles()[1..]).grid(storage.order());
            (per_word, words)
        }
    };
    let words: Vec<(Axis, usize)> = values
        .iter()
        .map(|&(axis, extent)| match axis == axes[axes.len() - 1] {
            true => (axis, extent / per_word),
            false => (axis, extent),
        })
        .collect();
    let misfit = |misfit: StorageMisfit| refused(misfit.to_string());
    let in_values = storage.physical(&values).map_err(misfit)?;
    let in_words = words_storage.physical(&words).map_err(misfit)?;
    let batch_dims: Vec<usize> = src.shape[..batches].to_vec();
    let config = |e| refused(format!("{e:?}"));
    // Batch dims stay plain and coarsest, each a whole run of what is finer.
    let metadata = |stored: &Geometry| {
        let mut run: usize = stored.shape().iter().product();
        let mut strides = vec![0; batches];
        for (b, &extent) in batch_dims.iter().enumerate().rev() {
            strides[b] = run;
            run *= extent;
        }
        let shape: Vec<usize> = batch_dims.iter().chain(stored.shape()).copied().collect();
        strides.extend_from_slice(stored.strides());
        let fragments: Vec<usize> = batch_dims
            .iter()
            .map(|_| 1)
            .chain(stored.tiling().fragments(axes.len()))
            .collect();
        Metadata::new(Shape::from(shape), Strides::from(strides))
            .with_tiling(Tiling::new(&fragments).map_err(config)?)
            .map_err(config)
    };
    let dst_values = metadata(&in_values)?;
    let dst_words = metadata(&in_words)?;
    let word_count: usize =
        words.iter().map(|&(_, e)| e).product::<usize>() * batch_dims.iter().product::<usize>();
    let handle = client.empty(word_count * dtype.size());
    let dst = TensorHandle::from_metadata(handle.clone(), dst_words, dtype);
    let mut src_words = src;
    for (d, stride) in src_words.strides.iter_mut().enumerate() {
        if d + 1 < rank {
            *stride /= per_word;
        }
    }
    src_words.shape[rank - 1] /= per_word;
    let tiles: Vec<(Axis, usize)> = words
        .iter()
        .map(|&(axis, _)| {
            let tile: usize = words_storage
                .tiles()
                .iter()
                .filter(|&&(a, _)| a == axis)
                .map(|&(_, count)| count)
                .product();
            (axis, tile)
        })
        .collect();
    relayout_launch(
        client,
        src_words,
        dst.binding(),
        dtype,
        &batch_dims,
        &words,
        &tiles,
    )?;
    Ok(TensorHandle::from_metadata(handle, dst_values, dtype))
}

/// Lay a storage-tiled tensor of plain values back out row-major, every logical dim in order.
///
/// # Errors
///
/// A source that is not storage-tiled.
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
    let logical = Metadata::new(src.shape.clone(), src.strides.clone())
        .with_tiling(src.tiling)
        .and_then(|metadata| metadata.logical_shape())
        .map_err(|e| refused(format!("{e:?}")))?;
    // Leading dims stored in one piece are batches; the rest are named, each dim's outermost
    // stored tile its extent over its grid's count, the grid being its coarsest piece, which a
    // tiling lists first.
    let fragments = src.tiling.fragments(logical.len());
    let batches = fragments.iter().take_while(|&&f| f == 1).count();
    let extents: Vec<(Axis, usize)> = (batches..logical.len())
        .map(|d| (Axis(d as u8), logical[d]))
        .collect();
    let tiles: Vec<(Axis, usize)> = extents
        .iter()
        .map(|&(axis, extent)| (axis, extent / src.shape[axis.0 as usize]))
        .collect();
    let dst = TensorHandle::empty(client, logical.clone(), dtype);
    relayout_launch(
        client,
        src,
        dst.clone().binding(),
        dtype,
        &logical[..batches],
        &extents,
        &tiles,
    )?;
    Ok(dst)
}

/// The one launch both directions share: a space over the named axes (and one cell of each
/// batch) cut by one level, the outermost stored tile, a cube of one plane per tile. Each side's
/// binding says whether it is the tiled one, which decodes the levels inside that tile.
fn relayout_launch(
    client: &Client,
    src: TensorBinding,
    dst: TensorBinding,
    dtype: ElemType,
    batches: &[usize],
    extents: &[(Axis, usize)],
    tiles: &[(Axis, usize)],
) -> Result<(), MatmulSetupError> {
    let named: Vec<Axis> = extents.iter().map(|&(axis, _)| axis).collect();
    // Batch axes the named ones leave free.
    let batch_axes: Vec<Axis> = (0..=u8::MAX)
        .map(Axis)
        .filter(|axis| !named.contains(axis))
        .take(batches.len())
        .collect();
    let batch: Vec<(Axis, usize)> = batch_axes
        .iter()
        .copied()
        .zip(batches.iter().copied())
        .filter(|&(_, b)| b > 1)
        .collect();
    let space = Space::new(&batch.iter().chain(extents).copied().collect::<Vec<_>>());
    let level = Levels::leaf(tiles)
        .cubes(&named)
        .batches(&batch.iter().map(|&(a, _)| a).collect::<Vec<_>>())
        .build()
        .remove(0);
    let partitioning = Partitioning::new(space, vec![level.clone()]);
    let cube_count = partitioning.cube_count();
    let cube_dim = CubeDim::new_1d(client.properties().hardware.plane_size_max);
    let concrete = partitioning.space().clone();
    let launch = Launcher::new(
        client,
        partitioning.all_dynamic(),
        &concrete,
        Grid::Stated {
            cube_count: cube_count.clone(),
            cube_dim,
        },
    )?;
    // A line runs along the innermost named axis when both sides end with it.
    let innermost = named[named.len() - 1];
    let (src_geometry, dst_geometry) = (Geometry::from(&src), Geometry::from(&dst));
    let (src_labels, dst_labels) = (src_geometry.labels(&named), dst_geometry.labels(&named));
    let v = match (src_labels.last(), dst_labels.last()) {
        (Some(&s), Some(&d)) if s == innermost && d == innermost => launch.vector_size(
            innermost,
            &[(&src_geometry, &src_labels), (&dst_geometry, &dst_labels)],
            dtype.size(),
        ),
        _ => 1,
    };
    let bind = |binding| {
        launch
            .arg(binding)
            .axes(&named)
            .batches(&batch_axes)
            .vectorize(v)
            .build()
    };
    let (s, d) = (bind(src)?, bind(dst)?);
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
    Ok(())
}
