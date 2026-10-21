use crate::microkernel::MatmulOperands;
use cubek_tile::layout::ElemAlignment;

/// Where each operand's buffer starts, how its rows are aligned over the order it is bound in, and
/// the tile it is stored in where it is: what a bulk copy descriptor is encoded from, which a work
/// item, being the same for every box, does not state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulBuffers {
    pub alignments: MatmulOperands<ElemAlignment>,
    pub start_bytes: MatmulOperands<u64>,
    pub stored_tiles: MatmulOperands<Option<MatmulStoredTile>>,
}

/// The tile an operand is stored in, over the axes it is bound along: `rows` of its rows' axis by
/// `columns` of `K`, each row of a tile one contiguous run, the tiles laid one after another. A
/// descriptor over the tiles lands a stage of whole tiles row by row, as it lands rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulStoredTile {
    pub rows: usize,
    pub columns: usize,
}
