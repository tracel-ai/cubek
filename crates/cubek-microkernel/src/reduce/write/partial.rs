use crate::reduce::{Accumulable, ReduceWorkItem, takes_partial_indices};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit storing its carrier as one of its rows' partials, as
/// [`UnitPartialsReduceMicrokernel`](crate::reduce::read::UnitPartialsReduceMicrokernel) loads it
/// back: the carrier's lines side by side, the indices beside them where it keeps any.
///
/// It is part of the cube's partial reduce, which hands it the carrier its cube met.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitPartialStoreMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
}

impl UnitPartialStoreMicrokernel {
    pub fn new(item: ReduceWorkItem) -> Self {
        Self { item }
    }
}

#[cube]
impl UnitPartialStoreMicrokernel {
    /// `acc` stored as partial `part` of `partials` and `partial_indices`
    /// (`unit_partial_store_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - `partial_indices` holds a tile exactly where the carrier keeps indices
    ///   ([`ReduceCarrier::carries_indices`](crate::reduce::ReduceCarrier::carries_indices)).
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where its Assumption is broken.
    pub fn store<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        acc: C,
        partials: &mut Tile,
        partial_indices: &ComptimeOption<Tile>,
        part: usize,
    ) {
        unit_partial_store_microkernel::<Acc, N, C>(
            acc,
            partials,
            partial_indices,
            part,
            comptime!(UnitPartialStoreMicrokernel { item: self.item }),
        )
    }
}

/// A partial is its slots' lines side by side.
#[cube]
pub(crate) fn unit_partial_store_microkernel<Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
    acc: C,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    part: usize,
    #[comptime] microkernel: UnitPartialStoreMicrokernel,
) {
    let item = microkernel.item;
    let slots = item.operation.written_per_row();
    let handed = partial_indices.is_some();
    comptime!(assert!(
        takes_partial_indices(item.operation.carrier(), handed),
        "UnitPartialStoreMicrokernel: {:?} takes its partials' indices exactly where it keeps any",
        item.operation.carrier()
    ));

    #[unroll]
    for slot in 0..slots {
        let line = part * slots + slot;
        partials
            .flat_mut::<Acc, N>()
            .write(line, C::value_line(&acc, slot));
        #[comptime]
        match partial_indices.clone() {
            ComptimeOption::Some(mut tile) => tile
                .flat_mut::<u32, N>()
                .write(line, C::index_line(&acc, slot)),
            ComptimeOption::None => {}
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitPartialMicrokernelLauncher;

    /// A partial of four rows a line, stored as it is.
    mod partial_across_one_value_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitPartialMicrokernelLauncher, Across, 1, 4, Plain, [
            sum: ReduceOperation::Sum,
            max: ReduceOperation::Max,
        ]);
    }

    /// Rows of one value, a value a line: a partial of one cell, three rows of it.
    mod partial_along_one_value_one_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitPartialMicrokernelLauncher, Along, 1, 1, Plain, [
            sum: ReduceOperation::Sum,
        ]);
    }
}
