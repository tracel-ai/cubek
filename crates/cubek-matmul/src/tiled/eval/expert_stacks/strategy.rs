use cubek_test_utils::CatalogEntry;

/// How the stack's words and block scales lie in memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum StackLayout {
    /// Rows padded as CUDA's and HIP's pitched allocator pads them when a weight is copied to the
    /// device ([`pitch`](super::benchmark::pitch)): what a stack kept in rows holds there.
    PaddedRows,
    /// The same rows back to back, as Metal's and wgpu's allocators leave them.
    DenseRows,
    /// Each row padded to the next 16 bytes, the widest load: the least padding that keeps every
    /// row's words and scales aligned for a vectorized read.
    AlignedRows,
    /// Tile order, in the tile and the order this device's writer lays a packed weight in
    /// ([`stored_tile`](crate::tiled::storage::stored_tile),
    /// [`tile_order_on`](crate::tiled::storage::tile_order_on)).
    TileOrder,
}

pub fn strategies() -> Vec<CatalogEntry<StackLayout>> {
    vec![
        CatalogEntry::new("padded_rows", "rows, pitched", StackLayout::PaddedRows),
        CatalogEntry::new("dense_rows", "rows, dense", StackLayout::DenseRows),
        CatalogEntry::new(
            "aligned_rows",
            "rows, 16-byte aligned",
            StackLayout::AlignedRows,
        ),
        CatalogEntry::new("tile_order", "tile order", StackLayout::TileOrder),
    ]
}
