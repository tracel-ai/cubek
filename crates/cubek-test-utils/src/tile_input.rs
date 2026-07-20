//! A clean, launchable [`Tile`] input for tests. Describe a tile as a [`Space`]
//! with a stack of `.split`/`.tile` levels and get a launchable tile: the buffer
//! is a plain `[grid…, tile…]` strided tensor (or `[grid…, level1…, …]` when
//! recursively tiled), presented in its logical space via an explicit
//! `tiled_view`. No `Tiler`, no semantic-view juggling.
#![allow(dead_code)]

use cubecl::std::tensor::{
    TensorHandle,
    layout::tiled_view::{TileSpec, TiledViewLaunch, TiledViewLayout},
};
use cubecl::{
    TestRuntime, bytes::Bytes, client::ComputeClient, prelude::CubePrimitive, prelude::TensorArg,
    quant::scheme::QuantScheme, zspace::Shape,
};
use cubek_tile::{Space, Storage};

use crate::{TestInput, TestInputBuilder};

/// A tile-shaped test input: the device buffer plus the logical [`Space`] it's
/// viewed in. The sub-tile sizes live in the buffer's trailing dims, so the view
/// reads them from there.
pub struct TileInput {
    handle: TensorHandle<TestRuntime>,
    space: Space,
    levels: usize,
}

impl TileInput {
    /// Start building a tile over `space`. Stack tiling levels coarse→fine with
    /// [`split`](TileInputBuilder::split) (by count) or
    /// [`tile`](TileInputBuilder::tile) (by element edge) — chain them for
    /// recursion, or [`untiled`](TileInputBuilder::untiled) for none — then a data
    /// finalizer ([`arange`](TileInputBuilder::arange) /
    /// [`zeros`](TileInputBuilder::zeros)).
    pub fn builder(client: &ComputeClient<TestRuntime>, space: Space) -> TileInputBuilder {
        TileInputBuilder {
            client: client.clone(),
            space,
            levels: None,
        }
    }

    /// Launch arg for this tile's view — the buffer seen in its logical space.
    /// Every logical axis is tiled (`num_tiled = space.rank()`), recursively for
    /// `levels` nested tile levels.
    pub fn view(&self) -> TiledViewLaunch<TestRuntime> {
        TiledViewLaunch::new_tensor::<TiledViewLayout>(
            self.handle.clone().binding().into_tensor_arg(),
            TileSpec {
                start_axis: 0,
                num_tiled: self.space.rank(),
                levels: self.levels,
            },
        )
    }

    /// Launch arg for this tile's view vectorized with line size `vector_size`
    /// along the innermost (contiguous) axis. Same buffer; only the metadata is
    /// reinterpreted in line units (innermost shape ÷ `S`, every other stride
    /// ÷ `S`), so a kernel reading `Vector<E, S>` lands on contiguous lines.
    /// `vector_size == 1` is exactly [`view`](Self::view).
    pub fn view_vectorized(&self, vector_size: usize) -> TiledViewLaunch<TestRuntime> {
        let shape = self.handle.shape();
        let strides = self.handle.strides();
        let inner = shape.len() - 1;
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s / vector_size } else { s })
            .collect();
        let new_strides: Vec<usize> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s } else { s / vector_size })
            .collect();
        let lined = TensorHandle::<TestRuntime>::new(
            self.handle.handle.clone(),
            new_shape,
            new_strides,
            self.handle.dtype,
        );
        TiledViewLaunch::new_tensor::<TiledViewLayout>(
            lined.binding().into_tensor_arg(),
            TileSpec {
                start_axis: 0,
                num_tiled: self.space.rank(),
                levels: self.levels,
            },
        )
    }

    /// Launch arg for this tile's raw global buffer — a plain [`TensorArg`] over the
    /// `[grid…, tile…]` buffer, optionally re-lined by `vector_size` along the inner
    /// axis (so a kernel reading `Vector<E, S>` lands on contiguous lines).
    /// `StridedTileArgLaunch::strided` wraps it in a `TiledViewLayout` at launch (this just produces the
    /// raw physical buffer/strides). `vector_size == 1` is the plain buffer.
    pub fn tensor_arg(&self, vector_size: usize) -> TensorArg<TestRuntime> {
        if vector_size <= 1 {
            return self.handle.clone().binding().into_tensor_arg();
        }
        let shape = self.handle.shape();
        let strides = self.handle.strides();
        let inner = shape.len() - 1;
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s / vector_size } else { s })
            .collect();
        let new_strides: Vec<usize> = strides
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == inner { s } else { s / vector_size })
            .collect();
        TensorHandle::<TestRuntime>::new(
            self.handle.handle.clone(),
            new_shape,
            new_strides,
            self.handle.dtype,
        )
        .binding()
        .into_tensor_arg()
    }

    /// The tensor's physical [`Storage`] — derived from the buffer's rank vs the
    /// logical space's rank, so the launch never hand-writes tile levels.
    pub fn storage(&self) -> Storage {
        Storage::of(self.handle.shape().len(), self.space.rank())
    }

    /// The semantic space the tile lives in.
    pub fn space(&self) -> Space {
        self.space.clone()
    }

    /// The device handle, for reading an output back.
    pub fn handle(&self) -> TensorHandle<TestRuntime> {
        self.handle.clone()
    }
}

/// One tiling level, added coarse→fine. [`Split`](TileLevel::Split) and
/// [`Tile`](TileLevel::Tile) are duals against the running tile edge: `Tile(e)`
/// sets the current tile to `e` elements; `Split(n)` divides it into `n`.
enum TileLevel {
    /// Divide the current tile into this many sub-tiles per axis.
    Split(Vec<usize>),
    /// Set the current tile to this many elements per axis.
    Tile(Vec<usize>),
}

/// Fluent builder for a [`TileInput`]: a [`Space`], a coarse→fine stack of tiling
/// levels (each a [`split`](Self::split) or [`tile`](Self::tile)), and a data
/// finalizer that fills the `[grid…, level…, finest…]` buffer.
pub struct TileInputBuilder {
    client: ComputeClient<TestRuntime>,
    space: Space,
    levels: Option<Vec<TileLevel>>,
}

impl TileInputBuilder {
    /// Divide the current tile into `counts[axis]` sub-tiles per axis — a finer
    /// level. Chain for recursion: `.split(&[4, 4]).split(&[2, 2])`.
    pub fn split(mut self, counts: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Split(counts.to_vec()));
        self
    }

    /// Set the current tile to `edges[axis]` elements per axis — a finer level.
    /// The dual of [`split`](Self::split) (it divides the current edge down to
    /// `edges`), so `.tile(&[16, 16]).tile(&[8, 8])` ≡ `.tile(&[16, 16]).split(&[2, 2])`.
    pub fn tile(mut self, edges: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Tile(edges.to_vec()));
        self
    }

    /// No sub-tiling: the buffer is the logical shape itself, row-major (zero tile
    /// levels — the view is the identity).
    pub fn untiled(mut self) -> Self {
        self.levels = Some(Vec::new());
        self
    }

    /// Arange `0, 1, 2, …` written straight onto the physical buffer; the
    /// `tiled_view` then presents it in logical coordinates.
    pub fn arange(self) -> TileInput {
        self.build(TestInputBuilder::arange)
    }

    /// All-zeros physical buffer — e.g. a matmul output.
    pub fn zeros(self) -> TileInput {
        self.build(TestInputBuilder::zeros)
    }

    /// Uniformly distributed random values in `[lo, hi)` range, seeded by `seed`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TileInput {
        self.build(|b: TestInputBuilder| b.uniform(seed, lo, hi))
    }

    /// Continue as a packed-u32 quantized input: a *container* step, like the tiling ones —
    /// the tile will be declared **in values** (its shape and strides count them) while the
    /// buffer holds the packed words, the binding convention the tile launch expects for a
    /// packed operand. Content comes from the quantized builder's own finalizer (e.g.
    /// [`arange`](QuantizedTileInputBuilder::arange)), which also mints the scales — a
    /// quantized tensor is one thing (data + scales + scheme). Untiled only: packed storage
    /// has no physically tiled layout.
    pub fn packed(self, scheme: &QuantScheme) -> QuantizedTileInputBuilder {
        let levels = self
            .levels
            .expect("TileInput: set .untiled() before .packed");
        assert!(
            levels.is_empty(),
            "TileInput::packed: a packed input is untiled"
        );
        QuantizedTileInputBuilder {
            client: self.client,
            space: self.space,
            scheme: *scheme,
        }
    }

    /// Build the `[grid…, level…, finest…]` device buffer, filled by `fill` (a
    /// `TestInput` finalizer like `arange`/`zeros`) in physical row-major order.
    /// Walking coarse→fine, each level becomes one block of `rank` dims and the
    /// leftover edge is the finest block — `(levels + 1) * rank` dims, the layout
    /// the `tiled_view` reads back.
    fn build(self, fill: impl FnOnce(TestInputBuilder) -> TestInput) -> TileInput {
        let levels = self
            .levels
            .expect("TileInput: set .split/.tile(...) or .untiled() before a finalizer");
        let rank = self.space.rank();

        let mut current: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.axis_at(i)))
            .collect();
        let mut blocks: Vec<Vec<usize>> = Vec::with_capacity(levels.len() + 1);
        for level in &levels {
            let (values, is_split) = match level {
                TileLevel::Split(values) => (values, true),
                TileLevel::Tile(values) => (values, false),
            };
            assert_eq!(
                values.len(),
                rank,
                "TileInput: a tile level needs one value per axis (rank {rank})"
            );
            let block: Vec<usize> = (0..rank)
                .map(|i| {
                    let (edge, value) = (current[i], values[i]);
                    assert!(
                        value != 0 && edge % value == 0,
                        "TileInput: tile value {value} does not divide the current edge {edge} on axis {i}"
                    );
                    let (dim, next) = if is_split {
                        (value, edge / value)
                    } else {
                        (edge / value, value)
                    };
                    current[i] = next;
                    dim
                })
                .collect();
            blocks.push(block);
        }
        blocks.push(current);

        let mut dims = Vec::with_capacity(blocks.len() * rank);
        for block in &blocks {
            dims.extend_from_slice(block);
        }
        let builder = TestInput::builder(self.client, Shape::from(dims));
        TileInput {
            handle: fill(builder).generate_without_host_data(),
            space: self.space,
            levels: levels.len(),
        }
    }
}

/// The quantized continuation of a [`TileInputBuilder`]: the container is fixed
/// (packed-u32, values-unit declaration, the scheme's block grid for scales); a content
/// finalizer fills it and mints the values tile and its scales together — a quantized
/// tensor is one thing (data, scales, scheme).
pub struct QuantizedTileInputBuilder {
    client: ComputeClient<TestRuntime>,
    space: Space,
    scheme: QuantScheme,
}

impl QuantizedTileInputBuilder {
    /// `arange`, wrapped into the scheme's representable range: values run
    /// `lo, lo+1, …, hi, lo, …`, so every representable value (and its sign extension)
    /// appears. Scales are a distinct-per-block ramp (`0.05 · (block + 1)`) on the scheme's
    /// block grid.
    pub fn arange(self) -> QuantizedTileInput {
        let rank = self.space.rank();
        let shape: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.axis_at(i)))
            .collect();

        let (lo, hi) = self.scheme.value.range();
        let (lo, hi) = (lo as i32, hi as i32);
        let span = hi - lo + 1;
        let q: Vec<i32> = (0..shape.iter().product())
            .map(|i| lo + (i as i32 % span))
            .collect();
        let words = crate::stubs::quant::pack_q_values(&q, &self.scheme);
        let handle = self.client.create(Bytes::from_elems(words));

        let block = crate::stubs::quant::block_dims(&self.scheme, &shape);
        let grid = crate::stubs::quant::scales_shape(&shape, &block);
        let scale_values: Vec<f32> = (0..grid.iter().product())
            .map(|g| 0.05 * (g + 1) as f32)
            .collect();
        let scales = TestInput::builder(self.client, Shape::from(grid))
            .custom(scale_values.clone())
            .generate_without_host_data();

        QuantizedTileInput {
            tile: TileInput {
                handle: TensorHandle::new_contiguous(
                    shape,
                    handle,
                    u32::as_type_native_unchecked().storage_type(),
                ),
                space: self.space,
                levels: 0,
            },
            scales,
            q,
            scale_values,
        }
    }
}

/// A packed-quantized input born whole — a quantized tensor is one thing (data, scales,
/// scheme), so the [`quantized builder`](QuantizedTileInputBuilder) mints the pair together,
/// plus the exact numbers behind both for host references.
pub struct QuantizedTileInput {
    pub tile: TileInput,
    scales: TensorHandle<TestRuntime>,
    /// The quant values, row-major in the logical shape.
    pub q: Vec<i32>,
    /// One scale per block, row-major over the scheme's block grid.
    pub scale_values: Vec<f32>,
}

impl QuantizedTileInput {
    /// Launch arg for the scales tensor.
    pub fn scales_arg(&self) -> TensorArg<TestRuntime> {
        self.scales.clone().binding().into_tensor_arg()
    }
}
