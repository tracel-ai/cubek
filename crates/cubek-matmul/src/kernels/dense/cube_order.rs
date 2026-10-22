use cubecl::tile::space::CubeOrder;

/// The orders a routine's settings list besides the grid's own: strips four boxes wide, along
/// either axis.
///
/// The grid starts its cubes along `m` first, so a wave of them covers a band of the output as
/// long as `m` has boxes and a few boxes wide along `n`: every cube of the wave reads its own rows
/// of the lhs, and the next wave reads them all again. Strips gather a wave into a patch a few boxes
/// wide, so the cubes running together share rows and columns in the last level cache.
///
/// The width is the one a kernel compiles whatever the problem: cubek distributes a last strip the
/// axis cuts short, so it is never fitted to a grid's count, which would compile one kernel per
/// count.
pub const CUBE_STRIPS: [CubeOrder; 2] = [CubeOrder::SwizzleRow(4), CubeOrder::SwizzleCol(4)];

/// How a setting's name spells its cube order: nothing for the grid's own, so the settings that
/// run in it keep their names.
pub fn cube_order_name(order: CubeOrder) -> String {
    match order {
        CubeOrder::RowMajor => String::new(),
        CubeOrder::SwizzleRow(width) => format!(" cubes=m-strips-{width}"),
        CubeOrder::SwizzleCol(width) => format!(" cubes=n-strips-{width}"),
    }
}
