//! How the register nest takes one load of its rhs: a run along the rhs's innermost axis, and,
//! where the rhs is stored in tiles, several of its columns side by side.

use cubecl::prelude::*;

use crate::{algebra::comptime_only, *};

/// One load of the rhs as the register nest splits it: `run` values along the rhs's innermost
/// axis, for each of `columns` columns along `across`.
///
/// A plain buffer, or one stored a word along `K` and nothing finer, loads one column. One stored
/// a word of 8 along `K` and then 4 columns loads `8 × 4` values, which the nest splits into the
/// four columns' runs of 8: one read where a plain buffer takes four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RhsLoad {
    /// Values of one column a load holds, along the rhs's innermost axis.
    pub run: usize,
    /// Columns one load holds side by side.
    pub columns: usize,
    /// The axis those columns run along, where there are several.
    pub across: Option<Axis>,
}

impl RhsLoad {
    /// How `load` sits in a rhs spanning `rhs`.
    ///
    /// # Panics
    ///
    /// A load spanning more than two axes, or two whose finest is not the rhs's innermost: the
    /// nest reads a run along that axis and the columns beside it, nothing else.
    pub(crate) fn of(load: &VectorTile, rhs: &Space) -> Self {
        let innermost = rhs.axis_at(rhs.rank() - 1);
        match *load.extents() {
            [(_, run)] => RhsLoad {
                run,
                columns: 1,
                across: None,
            },
            [(finest, run), (across, columns)] if finest == innermost => RhsLoad {
                run,
                columns,
                across: Some(across),
            },
            _ => panic!(
                "mm: the register nest reads a rhs load as a run along {innermost:?} and the \
                 columns beside it, and this rhs loads {:?}",
                load.extents()
            ),
        }
    }

    /// Refuse a load of several columns a block of `nr` columns over `axes` of `rhs` cannot
    /// split: the rhs must line along the contraction (`folded`), the load's columns must be the
    /// innermost axis of the rhs matrix's rows, so load `g` holds rows `g * columns ..` in order,
    /// and the block must hold whole loads.
    pub(crate) fn check(&self, rhs: &Space, axes: MatrixAxes, folded: bool, nr: usize) {
        let Some(across) = self.across else {
            return;
        };
        let rows: Vec<Axis> = (axes.row_split..axes.col_split)
            .map(|p| rhs.axis_at(p))
            .collect();
        assert!(
            folded && rows.last() == Some(&across) && nr.is_multiple_of(self.columns),
            "mm: a rhs load holds {} columns along {across:?}, which the register nest splits \
             only where the rhs lines along the contraction, {across:?} is the innermost of its \
             columns {rows:?}, and the block's {nr} columns hold whole loads",
            self.columns
        );
    }
}

comptime_only!(RhsLoad);

#[cube]
impl<T: Numeric> Tile<T> {
    /// This tile's load as the register nest takes it when this tile is the rhs.
    pub(crate) fn rhs_load(&self) -> comptime_type!(RhsLoad) {
        let load = self.vector_tile();
        comptime!(RhsLoad::of(&load, &self.place.space))
    }
}
