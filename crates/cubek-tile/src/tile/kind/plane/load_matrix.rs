//! The `ldmatrix` transport of the manual-mma encoding: a lane hands the instruction its row's
//! address, placed where the window's layout places it, so padded and swizzled stages alike are
//! read in 16-byte rows rather than a cell at a time.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    prelude::*,
};

use crate::*;

/// Bytes one row of an `ldmatrix` 8×8 matrix holds, which one lane addresses.
pub(super) const LDMATRIX_ROW_BYTES: usize = 16;

/// `ldmatrix` load: each lane hands the instruction the address of one 16-byte row of one of the
/// fragment's 8×8 matrices, and the instruction deals every lane its cells. Lane `l` addresses
/// row `l % 8` of matrix `l / 8`, and the matrices lie where the fragment's registers do
/// ([`MmaDefinition::position_of_nth`] of lane 0), so the rows a lane addresses are the rows the
/// manual load would read its cells from, taken 16 bytes at a time.
///
/// The address is the window's own placement of that row ([`Masked::line_slice`]): a padded or a
/// swizzled stage is read where its fill wrote it, which a base and a row stride could not say of
/// a swizzled one. The instruction transposes where the window's order is not the one the
/// fragment's register vectors run along: a col weight bound as `{n, k}` beside a `B` whose
/// vectors run along `k` needs none.
///
/// A row is a whole number of the window's lines, and the window lies in shared memory, which a
/// stage's 16-byte alignment ([`Memory::smem_aligned`]) keeps every row's address aligned in: the
/// manual-mma fragment load reads any other window a cell at a time.
#[cube]
pub(super) fn load_ldmatrix<T: Numeric, W: Size, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] edges: (usize, usize),
) {
    let width = src.vector_size();
    let elem_size = T::size().comptime();
    let row_cells = comptime!(LDMATRIX_ROW_BYTES / elem_size);
    let (rows, cols) = comptime!(edges);
    let col_major_window = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load: an ldmatrix load reads a row- or col-major window")
        }
    });
    let view = if comptime!(col_major_window) {
        src.fragment_matrix_packed::<W>(cols, rows)
    } else {
        src.fragment_matrix_packed::<W>(rows, cols)
    };

    let num_regs = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let vector_layout = def.vector_layout(ident);
    // Whether the instruction transposes each 8×8 matrix on its way to the registers.
    let instruction_transposes = comptime!(vector_layout != layout);
    // Lanes are dealt out eight to a matrix, wrapping where the fragment holds fewer than four.
    let lane = UNIT_POS_PLANE;
    let sub_lane = lane % 8;
    let nth_matrix = lane / 8 % comptime!(num_regs as u32);
    let (row, col) = def.position_of_nth(0, nth_matrix * comptime!(vector_size as u32), ident);
    // The lane's row runs along the window's lines, and the next lane's starts a line-row on.
    let (line_row, cell) = if comptime!(col_major_window) {
        (col + sub_lane, row)
    } else {
        (row + sub_lane, col)
    };
    let row_slice = view.line_slice(
        (line_row, cell / comptime!(width as u32)),
        (
            1u32.runtime(),
            comptime!((row_cells / width) as u32).runtime(),
        ),
    );
    let regs =
        def.load_matrix::<Vector<T, W>, N>(row_slice, ident, num_regs, instruction_transposes);
    #[unroll]
    for i in 0..num_regs {
        fragment[i] = regs[i];
    }
}
