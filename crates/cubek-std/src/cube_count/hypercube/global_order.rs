/// The zigzag a swizzled order walks, which is a walk's own arithmetic and lives with the
/// walk ([`cubek_tile::swizzle`]). Re-exported here because [`GlobalOrder`]'s swizzle arms
/// are its one caller in this crate.
pub use cubek_tile::swizzle;

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Describes the global traversal order as flattened cube position increases.
///
/// - `RowMajor`: standard row-first traversal
/// - `ColMajor`: standard column-first traversal
/// - `SwizzleCol(w)`: zigzag pattern down columns, with `w`-wide steps
/// - `SwizzleRow(w)`: zigzag pattern across rows, with `w`-wide steps
///
/// Special cases:
/// - `SwizzleCol(1)` is equivalent to `ColMajor`
/// - `SwizzleRow(1)` is equivalent to `RowMajor`
///
/// Swizzle modes may fail if their `w` does not divide the problem well.
#[allow(clippy::enum_variant_names)]
pub enum GlobalOrder {
    #[default]
    RowMajor,
    ColMajor,
    SwizzleRow(u32),
    SwizzleCol(u32),
}

impl GlobalOrder {
    /// Since they are equivalent but the latter form will skip some calculations,
    /// - `SwizzleColMajor(1)` becomes `ColMajor`
    /// - `SwizzleRowMajor(1)` becomes `RowMajor`
    pub fn canonicalize(self) -> Self {
        match self {
            GlobalOrder::SwizzleCol(1) => GlobalOrder::ColMajor,
            GlobalOrder::SwizzleRow(1) => GlobalOrder::RowMajor,
            _ => self,
        }
    }
}
