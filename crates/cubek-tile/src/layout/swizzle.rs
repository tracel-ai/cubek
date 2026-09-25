//! Where a [`RowChunks::Swizzled`] stage keeps each line of a block row ([`ChunkSwizzle`]).

use cubecl::prelude::*;

use crate::*;

/// The XOR swizzle a [`RowChunks::Swizzled`] stage resolves to, once it knows its physical extents
/// and how many bytes a line holds.
///
/// A block row is cut into chunks of `lines_per_chunk` lines, and the chunk a line sits in is XORed
/// with a key read off its row. Rows sharing one bank period take one key, and the `keys` periods
/// that follow take the others, so the rows a fragment load reads at once start in distinct chunks
/// of the period and never share a bank. An XOR is its own inverse and only moves a chunk within
/// its row: a line's place inside its chunk, its row, and every digit above the block are kept.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ChunkSwizzle {
    /// The physical axis holding a block's rows.
    row_axis: usize,
    /// The physical axis holding a block row's lines, the innermost.
    line_axis: usize,
    /// Lines one chunk holds.
    lines_per_chunk: usize,
    /// Consecutive rows taking one key: the rows one bank period holds.
    rows_per_key: usize,
    /// How many keys the rows cycle through, a power of two: a row's chunks, at most a bank
    /// period's.
    keys: usize,
}

impl ChunkSwizzle {
    /// Bytes one pass over every bank covers: 32 banks of four bytes, on every GPU this addresses.
    pub(crate) const BANK_PERIOD_BYTES: usize = 128;

    /// The swizzle of a stage whose physical line extents are `extents`, the last two being its
    /// innermost block's rows and a row's lines, each line `line` long. `None` where a row holds
    /// one chunk or a whole bank period's worth per chunk: there is nothing to permute.
    pub(crate) fn new(extents: &[usize], line: LineBytes) -> Option<Self> {
        let LineBytes(line_bytes) = line;
        let rank = extents.len();
        assert!(
            rank >= 2,
            "ChunkSwizzle: a swizzled stage permutes the chunks of a row, and a rank-{rank} stage \
             has no rows"
        );
        assert!(
            line_bytes.is_power_of_two(),
            "ChunkSwizzle: a {line_bytes}-byte line does not tile a chunk"
        );
        let chunk_bytes = RowChunks::CHUNK_BYTES.max(line_bytes);
        let lines_per_chunk = chunk_bytes / line_bytes;
        let row_lines = extents[rank - 1];
        if row_lines * line_bytes <= chunk_bytes {
            return None;
        }
        assert!(
            row_lines.is_multiple_of(lines_per_chunk)
                && (row_lines / lines_per_chunk).is_power_of_two(),
            "ChunkSwizzle: a row of {row_lines} {line_bytes}-byte lines is not a power of two of \
             {chunk_bytes}-byte chunks, which an XOR permutes"
        );
        let chunks = row_lines / lines_per_chunk;
        let period_chunks = Self::BANK_PERIOD_BYTES / chunk_bytes;
        let keys = chunks.min(period_chunks);
        if keys < 2 {
            return None;
        }
        Some(Self {
            row_axis: rank - 2,
            line_axis: rank - 1,
            lines_per_chunk,
            rows_per_key: (period_chunks / chunks).max(1),
            keys,
        })
    }

    /// The physical axis holding a block's rows, whose digit keys the swizzle.
    pub(crate) fn row_axis(&self) -> usize {
        self.row_axis
    }

    /// The physical axis holding a block row's lines, whose digit the swizzle moves.
    pub(crate) fn line_axis(&self) -> usize {
        self.line_axis
    }

    /// The line `line` of row `row` is kept at, on the host: [`swizzled_line`]'s twin, which the
    /// tests hold to its properties.
    #[cfg(test)]
    fn line_of(&self, line: usize, row: usize) -> usize {
        line ^ ((row / self.rows_per_key) % self.keys * self.lines_per_chunk)
    }
}

/// The line `line` of block row `row` is kept at under `swizzle`; also the line kept at `line`,
/// the XOR being its own inverse.
#[cube]
pub(crate) fn swizzled_line(#[comptime] swizzle: ChunkSwizzle, line: u32, row: u32) -> u32 {
    let key = row
        .divided_by(comptime!(swizzle.rows_per_key as u32).runtime())
        .remainder(comptime!(swizzle.keys as u32).runtime());
    line ^ key.times(comptime!(swizzle.lines_per_chunk as u32).runtime())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte a line starts at, within its bank period.
    fn bank_offset(
        swizzle: &ChunkSwizzle,
        line_bytes: usize,
        row_lines: usize,
        row: usize,
    ) -> usize {
        let line = swizzle.line_of(0, row);
        (row * row_lines + line) * line_bytes % ChunkSwizzle::BANK_PERIOD_BYTES
    }

    /// A 64-byte row, 8-wide f16 lines (the cmma `16×32` block): the eight rows one pass of eight
    /// lanes reads start on eight distinct 16-byte slots of the bank period.
    #[test]
    fn eight_rows_of_a_64_byte_block_start_on_distinct_banks() {
        let swizzle = ChunkSwizzle::new(&[2, 16, 4], LineBytes(16)).unwrap();
        let mut slots: Vec<usize> = (0..8)
            .map(|row| bank_offset(&swizzle, 16, 4, row))
            .collect();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 8);
    }

    /// A 32-byte row (the mma `16×16` block) and a 256-byte one, and rows of 8- and 4-byte lines:
    /// the same, whatever the row holds. Every case permutes: none is a row of one chunk.
    #[test]
    fn rows_short_and_long_start_on_distinct_banks() {
        for (row_lines, line_bytes) in [(2, 16), (16, 16), (8, 8), (16, 4)] {
            let swizzle = ChunkSwizzle::new(&[16, row_lines], LineBytes(line_bytes))
                .expect("a row of several chunks is swizzled");
            let rows = ChunkSwizzle::BANK_PERIOD_BYTES / RowChunks::CHUNK_BYTES;
            let mut slots: Vec<usize> = (0..rows)
                .map(|row| {
                    bank_offset(&swizzle, line_bytes, row_lines, row) / RowChunks::CHUNK_BYTES
                })
                .collect();
            slots.sort();
            slots.dedup();
            assert_eq!(slots.len(), rows, "{row_lines} lines of {line_bytes} bytes");
        }
    }

    /// Every row's lines are a permutation of themselves, and a line keeps its place inside its
    /// chunk: a 4-byte line moves with the three beside it.
    #[test]
    fn a_row_is_permuted_within_itself_a_chunk_at_a_time() {
        let swizzle = ChunkSwizzle::new(&[16, 16], LineBytes(4)).unwrap();
        for row in 0..16 {
            let mut lines: Vec<usize> = (0..16).map(|line| swizzle.line_of(line, row)).collect();
            for line in 0..16 {
                assert_eq!(swizzle.line_of(line, row) % 4, line % 4);
                assert_eq!(swizzle.line_of(swizzle.line_of(line, row), row), line);
            }
            lines.sort();
            assert_eq!(lines, (0..16).collect::<Vec<_>>());
        }
    }

    /// A row of one chunk has nothing to permute, and neither has a row whose every chunk is a
    /// whole bank period.
    #[test]
    fn a_row_of_one_chunk_is_left_in_order() {
        assert!(ChunkSwizzle::new(&[16, 1], LineBytes(16)).is_none());
        assert!(ChunkSwizzle::new(&[16, 2], LineBytes(8)).is_none());
        assert!(ChunkSwizzle::new(&[16, 4], LineBytes(128)).is_none());
    }
}
