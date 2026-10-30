//! Storing a tensor in storage tiles, and laying it back out.
//!
//! A storage-tiled tensor is stored as the [`StoragePartitioning`] it was stored in states: its
//! tiles, finest first, then the grid of tiles. Its binding carries how many pieces each dim is
//! stored in (its `tiling`) and the order they follow one another in (its strides).
//!
//! Tiling is a relayout on the tile DSL: a space over the named axes with one level, the
//! outermost stored tile, one cube per tile. Whichever side is storage-tiled is read or written
//! through its layout, however many levels lie inside that tile.

use cubecl::tile::{
    Geometry, Launcher, Level, Levels, Partitioning, Space, TileArg, kind::Field, launch::Grid,
    launch::Refusal,
};
use cubecl::{
    client::Client,
    ir::ElemType,
    prelude::*,
    std::tensor::{TensorHandle, layout::CoordsDyn},
    zspace::{Shape, Strides, Tiling, metadata::Metadata},
};

use crate::definition::MatmulSetupError;

/// `dst = src` over their logical cells, one cube per region of `level`, the cube's units
/// striding the region's lines. A side inside its storage tile is a run; the other walks its
/// layout.
#[cube(launch)]
fn relayout<E: Numeric, V: Size>(
    src: &TileArg<'_>,
    dst: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(&space);
    let dst = dst.tile(&space);
    let rank = comptime!(space.space().rank());
    for cube in space.over(&level) {
        let s = src.at(&cube);
        let mut d = dst.at(&cube);
        let r = s.view::<E, V>();
        let mut w = d.view_mut::<E, V>();
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
pub use cubecl::tile::{
    Axis,
    layout::{StorageLevels, StoragePartitioning},
};

/// How a tensor of one shape lies stored as a [`StoragePartitioning`] states: its metadata in
/// values and in the words a packed one moves as, the relayout of its words into storage tiles,
/// and the words it allocates.
struct StoredLayout {
    values: Metadata,
    words: Metadata,
    relayout: Relayout,
    word_count: usize,
    per_word: usize,
}

impl StoredLayout {
    /// The layout of a tensor of `shape`, its trailing dims named by `axes` (the dims ahead stay
    /// plain and coarsest, one batch each), stored as `storage` states, `packed` saying the values
    /// sit `field.per_word()` to a `u32` along the last of `axes`, so its `dtype` is `u32`.
    ///
    /// # Errors
    ///
    /// A shape of fewer dims than the axes it is stated over, a packed statement over no axes, of
    /// words that are not `u32`, or whose word is not its finest stored tile, or a partitioning
    /// that does not close the tensor in whole tiles or is deeper than a tiling records.
    #[allow(clippy::result_large_err)]
    fn new(
        shape: &[usize],
        axes: &[Axis],
        dtype: ElemType,
        packed: Option<Field>,
        storage: &StoragePartitioning,
        refused: impl Fn(String) -> MatmulSetupError,
    ) -> Result<Self, MatmulSetupError> {
        let rank = shape.len();
        let Some(batches) = rank.checked_sub(axes.len()) else {
            return Err(refused(format!(
                "a shape of {rank} dims is stated over {} axes",
                axes.len()
            )));
        };
        let values: Vec<(Axis, usize)> = axes
            .iter()
            .copied()
            .zip(shape[batches..].iter().copied())
            .collect();
        // A packed tensor is relaid out as its words: the same statement with the word dropped,
        // the packed axis counted in words.
        let (per_word, words_storage) = match packed {
            None => (1, storage.clone()),
            Some(field) => {
                let Some(&packed_axis) = axes.last() else {
                    return Err(refused(
                        "a packed tensor names the axis it packs along".into(),
                    ));
                };
                if dtype != u32::elem_type_native() {
                    return Err(refused(format!(
                        "a {field:?} tensor is stored in u32 words, not {dtype:?}"
                    )));
                }
                let per_word = field.per_word();
                if storage.tiles().first() != Some(&(packed_axis, per_word)) {
                    return Err(refused(format!(
                        "a {field:?} word is {per_word} values along {packed_axis:?}, so it is \
                         the finest stored tile; the statement starts with {:?}",
                        storage.tiles().first()
                    )));
                }
                let words = StorageLevels::new(&storage.tiles()[1..]).grid(storage.order());
                (per_word, words)
            }
        };
        let word_extents: Vec<(Axis, usize)> = values
            .iter()
            .map(|&(axis, extent)| match Some(&axis) == axes.last() {
                true => (axis, extent / per_word),
                false => (axis, extent),
            })
            .collect();
        let misfit = |misfit: Refusal| refused(misfit.to_string());
        let in_values = storage.physical(&values).map_err(misfit)?;
        let in_words = words_storage.physical(&word_extents).map_err(misfit)?;
        let batch_dims: Vec<usize> = shape[..batches].to_vec();
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
        let tiles: Vec<(Axis, usize)> = word_extents
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
        let word_count = word_extents.iter().map(|&(_, e)| e).product::<usize>()
            * batch_dims.iter().product::<usize>();
        Ok(Self {
            values: metadata(&in_values)?,
            words: metadata(&in_words)?,
            relayout: Relayout {
                batches: batch_dims,
                extents: word_extents,
                tiles,
            },
            word_count,
            per_word,
        })
    }
}

/// A fresh storage-tiled tensor of `shape`, its trailing dims named by `axes`, stored as `storage`
/// states and as [`tile`] would store a tensor of that shape: the buffer allocated, its contents
/// unwritten. What a kernel writing its output straight into storage tiles is handed, so no
/// relayout of a tensor in rows follows it. `packed` and the result's units are [`tile`]'s.
///
/// # Errors
///
/// As [`tile`], but for a source: a shape of fewer dims than its axes, a packed statement over
/// no axes, of words that are not `u32`, or whose word is not its finest stored tile, or a
/// partitioning that does not close the tensor in whole tiles or is deeper than a tiling records.
#[allow(clippy::result_large_err)]
pub fn empty(
    client: &Client,
    shape: &[usize],
    axes: &[Axis],
    dtype: ElemType,
    packed: Option<Field>,
    storage: StoragePartitioning,
) -> Result<TensorHandle, MatmulSetupError> {
    let refused = |why: String| MatmulSetupError::InvalidConfig(Box::new(format!("empty: {why}")));
    let layout = StoredLayout::new(shape, axes, dtype, packed, &storage, refused)?;
    let handle = client.empty(layout.word_count * dtype.size());
    Ok(TensorHandle::from_metadata(handle, layout.values, dtype))
}

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
/// A source already storage-tiled or of fewer dims than its axes, a packed source over no axes,
/// of words that are not `u32`, whose word is not the finest stored tile or whose last named dim
/// is not its contiguous one, or a partitioning that does not close the tensor in whole tiles or
/// is deeper than a tiling records.
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
    let layout = StoredLayout::new(&src.shape, axes, dtype, packed, &storage, refused)?;
    // The layout refuses a source of fewer dims than its axes, and a packed one over no axes.
    let rank = src.shape.len();
    if packed.is_some() && src.strides[rank - 1] != 1 {
        return Err(refused(format!(
            "the words pack along {:?}, which is not the source's contiguous dim",
            axes[axes.len() - 1]
        )));
    }
    let handle = client.empty(layout.word_count * dtype.size());
    let dst = TensorHandle::from_metadata(handle.clone(), layout.words, dtype);
    let mut src_words = src;
    for (d, stride) in src_words.strides.iter_mut().enumerate() {
        if d + 1 < rank {
            *stride /= layout.per_word;
        }
    }
    src_words.shape[rank - 1] /= layout.per_word;
    layout
        .relayout
        .launch(client, src_words, dst.binding(), dtype)?;
    Ok(TensorHandle::from_metadata(handle, layout.values, dtype))
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
    let relayout = Relayout {
        batches: logical[..batches].to_vec(),
        extents,
        tiles,
    };
    let dst = TensorHandle::empty(client, logical, dtype);
    relayout.launch(client, src, dst.clone().binding(), dtype)?;
    Ok(dst)
}

/// The relayout both directions share: a space over the named axes (and one cell of each batch)
/// cut by one level, the outermost stored tile, a cube of one plane per tile.
struct Relayout {
    /// The batch dims ahead of the named axes, plain and coarsest.
    batches: Vec<usize>,
    /// The named axes and their extents.
    extents: Vec<(Axis, usize)>,
    /// The outermost stored tile along each named axis.
    tiles: Vec<(Axis, usize)>,
}

impl Relayout {
    /// Launch `dst = src`. Each side's binding says whether it is the tiled one, which decodes the
    /// levels inside that tile.
    fn launch(
        &self,
        client: &Client,
        src: TensorBinding,
        dst: TensorBinding,
        dtype: ElemType,
    ) -> Result<(), MatmulSetupError> {
        let Self {
            batches,
            extents,
            tiles,
        } = self;
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
                .arg(binding, dtype)
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
}
