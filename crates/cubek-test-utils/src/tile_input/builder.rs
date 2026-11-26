use cubecl::tile::Space;
use cubecl::{client::Client, quant::scheme::QuantScheme, zspace::Shape};

use crate::{QuantizedTileInputBuilder, TestInput, TestInputBuilder, TileInput};

/// One tiling level, added coarse→fine. [`Split`](TileLevel::Split) and
/// [`Tile`](TileLevel::Tile) are duals against the running tile edge: `Tile(e)`
/// sets the current tile to `e` elements; `Split(n)` divides it into `n`.
pub(super) enum TileLevel {
    /// Divide the current tile into this many sub-tiles per dimension.
    Split(Vec<usize>),
    /// Set the current tile to this many elements per dimension.
    Tile(Vec<usize>),
}

/// Fluent builder for a [`TileInput`]: a [`Space`], a coarse→fine stack of tiling
/// levels (each a [`split`](Self::split) or [`tile`](Self::tile)), and a data
/// finalizer that fills the `[grid…, level…, finest…]` buffer.
pub struct TileInputBuilder {
    pub(super) client: Client,
    pub(super) space: Space,
    pub(super) levels: Option<Vec<TileLevel>>,
}

impl TileInputBuilder {
    /// Divide the current tile into `counts[dimension]` sub-tiles per dimension: a finer
    /// level. Chain for recursion: `.split(&[4, 4]).split(&[2, 2])`.
    pub fn split(mut self, counts: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Split(counts.to_vec()));
        self
    }

    /// Set the current tile to `edges[dimension]` elements per dimension: a finer level.
    /// The dual of [`split`](Self::split) (it divides the current edge down to
    /// `edges`), so `.tile(&[16, 16]).tile(&[8, 8])` ≡ `.tile(&[16, 16]).split(&[2, 2])`.
    pub fn tile(mut self, edges: &[usize]) -> Self {
        self.levels
            .get_or_insert_with(Vec::new)
            .push(TileLevel::Tile(edges.to_vec()));
        self
    }

    /// No sub-tiling: the buffer is the logical shape itself, row-major (zero tile
    /// levels: the view is the identity).
    pub fn untiled(mut self) -> Self {
        self.levels = Some(Vec::new());
        self
    }

    /// Arange `0, 1, 2, …` written straight onto the physical buffer; the
    /// `tiled_view` then presents it in logical coordinates.
    pub fn arange(self) -> TileInput {
        self.build(TestInputBuilder::arange)
    }

    /// All-zeros physical buffer: e.g. a matmul output.
    pub fn zeros(self) -> TileInput {
        self.build(TestInputBuilder::zeros)
    }

    /// Uniformly distributed random values in `[lo, hi)` range, seeded by `seed`.
    pub fn uniform(self, seed: u64, lo: f32, hi: f32) -> TileInput {
        self.build(|b: TestInputBuilder| b.uniform(seed, lo, hi))
    }

    /// Continue as a packed-u32 quantized input: a *container* step, like the tiling ones.
    /// the tile will be declared **in values** (its shape and strides count them) while the
    /// buffer holds the packed words, the binding convention the tile launch expects for a
    /// packed operand. Content comes from the quantized builder's own finalizer (e.g.
    /// [`arange`](QuantizedTileInputBuilder::arange)), which also mints the scales: a
    /// quantized tensor is one thing (data + scales + scheme). Untiled only: packed storage
    /// has no physically tiled layout.
    ///
    /// The launch binds the three as one operand: [`values_arg`] under its codebook
    /// ([`table_arg`]) and its scales ([`scales_arg`]), and the kernel states the decode where it
    /// happens.
    ///
    /// [`values_arg`]: crate::QuantizedTileInput::values_arg
    /// [`scales_arg`]: crate::QuantizedTileInput::scales_arg
    /// [`table_arg`]: crate::QuantizedTileInput::table_arg
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
    /// leftover edge is the finest block: `(levels + 1) * rank` dims, the layout
    /// the `tiled_view` reads back.
    fn build(self, fill: impl FnOnce(TestInputBuilder) -> TestInput) -> TileInput {
        let levels = self
            .levels
            .expect("TileInput: set .split/.tile(...) or .untiled() before a finalizer");
        let rank = self.space.rank();

        let mut current: Vec<usize> = (0..rank)
            .map(|i| self.space.extent(self.space.dimension_at(i)))
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
                "TileInput: a tile level needs one value per dimension (rank {rank})"
            );
            let block: Vec<usize> = (0..rank)
                .map(|i| {
                    let (edge, value) = (current[i], values[i]);
                    assert!(
                        value != 0 && edge % value == 0,
                        "TileInput: tile value {value} does not divide the current edge {edge} on dimension {i}"
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
        TileInput::new(
            fill(builder).generate_without_host_data(),
            self.space,
            levels.len(),
        )
    }
}
