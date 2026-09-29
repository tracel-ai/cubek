//! Where a stage keeps each line of a block row ([`RowArrangement`]): the form [`RowChunks`]
//! resolves to once the stage knows its extents and its line.

use crate::*;

/// Bytes one physical line of a buffer holds: what a stage's block rows are placed by, which is
/// not its vector size once a packed line stores several values a word.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct LineBytes(pub(crate) usize);

/// Where a stage keeps each line of its innermost block's rows: what a [`RowChunks`] request
/// resolves to for a stage's physical extents and line size, and what every read and fill of the
/// stage addresses through.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RowArrangement {
    /// Each row one run, the next right after it: line `i` of the stage sits at offset `i`. Every
    /// buffer that is not a stage asking otherwise, and a stage whose rows are one chunk each,
    /// which already start on distinct banks.
    InOrder,
    /// In order, each row's chunks permuted ([`ChunkSwizzle`]).
    Swizzled(ChunkSwizzle),
    /// Each row followed by `lines` lines nothing reads: one chunk
    /// ([`RowChunks::CHUNK_BYTES`]), or one line where a line is wider, so a row starts that much
    /// further round the banks than the row before it. Only the stage's strides and its size
    /// change: a reader that addresses rows by a stride reads it as it reads one in order.
    Padded { lines: usize },
}

impl RowArrangement {
    /// Where the rows of a stage whose physical line extents are `extents` (the last two its
    /// innermost block's rows and a row's lines), each line `line` long, are kept under `chunks`.
    pub(crate) fn new(chunks: RowChunks, extents: &[usize], line: LineBytes) -> Self {
        let row_bytes = extents.last().copied().unwrap_or(0) * line.0;
        let one_chunk = row_bytes <= RowChunks::CHUNK_BYTES.max(line.0);
        match (chunks, one_chunk) {
            (RowChunks::InOrder, _) | (_, true) => Self::InOrder,
            (RowChunks::Swizzled, false) => match ChunkSwizzle::new(extents, line) {
                Some(swizzle) => Self::Swizzled(swizzle),
                None => Self::InOrder,
            },
            (RowChunks::Padded, false) => Self::Padded {
                lines: RowChunks::CHUNK_BYTES.div_ceil(line.0),
            },
        }
    }

    /// Whether a row is followed by padding, so line `i` of the stage no longer sits at offset
    /// `i` and a fill writes it at its pitched offset ([`BufferLayout::line_offset`]).
    pub(crate) fn is_pitched(&self) -> bool {
        match self {
            Self::InOrder | Self::Swizzled(_) => false,
            Self::Padded { .. } => true,
        }
    }

    /// Whether a row is one run whose next row sits a stride further on: what a reader that
    /// addresses rows by a pointer and a stride (`cmma`, a raw window) needs.
    pub(crate) fn rows_are_runs(&self) -> bool {
        match self {
            Self::InOrder | Self::Padded { .. } => true,
            Self::Swizzled(_) => false,
        }
    }

    /// Refuses a stage whose rows are not in order to `reader`, which takes the stage as one dense
    /// run of rows (a TMA box, a window addressed as one run); `why` says what it needs.
    ///
    /// # Panics
    ///
    /// Where the rows are swizzled or padded.
    pub(crate) fn assert_in_order(&self, reader: &str, why: &str) {
        assert!(
            *self == Self::InOrder,
            "{reader}: {why}, so a {self:?} stage cannot be read by it; stage it RowChunks::InOrder"
        );
    }

    /// Refuses a stage whose rows are not runs a stride apart to `reader`, which addresses rows
    /// by a pointer and a stride (`cmma`, a raw window).
    ///
    /// # Panics
    ///
    /// Where the rows are swizzled.
    pub(crate) fn assert_rows_are_runs(&self, reader: &str) {
        assert!(
            self.rows_are_runs(),
            "{reader}: a swizzled stage keeps a row's chunks out of order, so a fragment API \
             reading rows off a pointer and a stride (cmma) cannot read it; stage it \
             RowChunks::InOrder or RowChunks::Padded"
        );
    }

    /// Lines of padding after each row.
    pub(crate) fn padding(&self) -> usize {
        match self {
            Self::InOrder | Self::Swizzled(_) => 0,
            Self::Padded { lines } => *lines,
        }
    }

    /// The swizzle that moves the digits of physical axis `axis`, where one does: a swizzled
    /// stage's line axis.
    pub(crate) fn swizzle_along(&self, axis: usize) -> Option<ChunkSwizzle> {
        match self {
            Self::Swizzled(swizzle) if swizzle.line_axis() == axis => Some(*swizzle),
            Self::InOrder | Self::Swizzled(_) | Self::Padded { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swizzled() -> RowArrangement {
        RowArrangement::new(RowChunks::Swizzled, &[16, 4], LineBytes(16))
    }

    fn padded() -> RowArrangement {
        RowArrangement::new(RowChunks::Padded, &[16, 4], LineBytes(16))
    }

    /// A TMA box and a window read as one run take rows in order only.
    #[test]
    #[should_panic(expected = "stage it RowChunks::InOrder")]
    fn a_dense_reader_refuses_a_swizzled_stage() {
        RowArrangement::InOrder.assert_in_order("reader", "why");
        swizzled().assert_in_order("reader", "why");
    }

    #[test]
    #[should_panic(expected = "stage it RowChunks::InOrder")]
    fn a_dense_reader_refuses_a_padded_stage() {
        padded().assert_in_order("reader", "why");
    }

    /// A fragment API reading rows off a pointer and a stride takes a padded row, never a
    /// swizzled one.
    #[test]
    #[should_panic(expected = "a swizzled stage keeps a row's chunks out of order")]
    fn a_fragment_api_refuses_a_swizzled_stage() {
        RowArrangement::InOrder.assert_rows_are_runs("cmma");
        padded().assert_rows_are_runs("cmma");
        swizzled().assert_rows_are_runs("cmma");
    }

    /// Rows of four 16-byte lines: kept in order, swizzled, or padded by one line; a row of one
    /// line is kept in order whatever was asked, a narrow line pads by a whole chunk of them, and
    /// a line wider than a chunk pads by one line.
    #[test]
    fn a_request_resolves_against_the_rows_it_lays_down() {
        let extents = [2, 16, 4];
        assert_eq!(
            RowArrangement::new(RowChunks::InOrder, &extents, LineBytes(16)),
            RowArrangement::InOrder
        );
        assert!(matches!(
            RowArrangement::new(RowChunks::Swizzled, &extents, LineBytes(16)),
            RowArrangement::Swizzled(_)
        ));
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &extents, LineBytes(16)),
            RowArrangement::Padded { lines: 1 }
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 1], LineBytes(16)),
            RowArrangement::InOrder
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 16], LineBytes(4)),
            RowArrangement::Padded { lines: 4 }
        );
        assert_eq!(
            RowArrangement::new(RowChunks::Padded, &[16, 4], LineBytes(32)),
            RowArrangement::Padded { lines: 1 }
        );
    }
}
