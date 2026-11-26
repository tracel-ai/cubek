use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem, takes_partial_indices};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit reducing the partials several cubes stored of its rows
/// ([`UnitPartialStoreMicrokernel`](crate::reduce::write::UnitPartialStoreMicrokernel)): each
/// partial loaded back into its carrier, the carriers combined and the rows written.
///
/// The hypercube's last cube to finish runs it on one unit.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitPartialsReduceMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// As [`UnitCarrierWriteMicrokernel`](crate::reduce::write::UnitCarrierWriteMicrokernel)'s.
    #[cube(comptime)]
    row_lanes: usize,
}

impl UnitPartialsReduceMicrokernel {
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }
}

#[cube]
impl UnitPartialsReduceMicrokernel {
    /// The first `parts` partials of `partials` and `partial_indices` reduced into `out` and
    /// `indices`, `values` the values each of the rows holds (`unit_partials_reduce_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - `indices` holds a tile exactly where the item writes the indices beside the values
    ///   ([`ReduceWrites::Both`](crate::reduce::ReduceWrites::Both)).
    /// - `partial_indices` holds a tile exactly where the carrier keeps indices
    ///   ([`ReduceCarrier::carries_indices`](crate::reduce::ReduceCarrier::carries_indices)).
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where an Assumption, or one of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s, is broken.
    pub fn reduce<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        partials: &Tile,
        partial_indices: &ComptimeOption<Tile>,
        parts: usize,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_partials_reduce_microkernel::<Out, Idx, Acc, N, C>(
            partials,
            partial_indices,
            parts,
            out,
            indices,
            values,
            comptime!(UnitPartialsReduceMicrokernel {
                item: self.item,
                row_lanes: self.row_lanes
            }),
        )
    }
}

/// The carriers combine in the order their partials lie.
#[cube]
pub(crate) fn unit_partials_reduce_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    partials: &Tile,
    partial_indices: &ComptimeOption<Tile>,
    parts: usize,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitPartialsReduceMicrokernel,
) {
    let item = microkernel.item;
    let monoid = item.operation.monoid();
    let slots = item.operation.written_per_row();
    let handed = partial_indices.is_some();
    comptime!(assert!(
        takes_partial_indices(item.operation.carrier(), handed),
        "UnitPartialsReduceMicrokernel: {:?} takes its partials' indices exactly where it keeps any",
        item.operation.carrier()
    ));

    let value_lines = partials.flat::<Acc, N>();
    let mut merged = C::identity(monoid);
    for part in 0..parts {
        let mut stored = C::identity(monoid);
        #[unroll]
        for slot in 0..slots {
            let line = part * slots + slot;
            #[comptime]
            let index = match partial_indices {
                ComptimeOption::Some(tile) => tile.flat::<u32, N>().read(line),
                ComptimeOption::None => Vector::new(0u32),
            };
            C::set_line(&mut stored, slot, value_lines.read(line), index);
        }
        merged = C::combine(merged, stored, monoid);
    }

    unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
        merged,
        out,
        indices,
        values,
        comptime!(UnitCarrierWriteMicrokernel::new(
            item,
            microkernel.row_lanes
        )),
    );
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitPartialsMicrokernelLauncher;

    /// Nine partials of four lanes of one row each.
    mod partials_along_36_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitPartialsMicrokernelLauncher, Along, 36, 4, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }

    /// Thirty-seven partials, each lane its own row.
    mod partials_across_37_long_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitPartialsMicrokernelLauncher, Across, 37, 4, Plain, [
            sum: ReduceOperation::Sum,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
        ]);
    }
}
