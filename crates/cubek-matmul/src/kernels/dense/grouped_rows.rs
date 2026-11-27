use crate::kernels::MatmulPartitioning;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Dimension, Levels, space::CubeOrder};
use cubek_microkernel::matmul::MatmulDestination;

/// The flat rows of a product whose table ends row groups
/// ([`MatmulIndex::GroupedRows`](crate::kernels::MatmulIndex::GroupedRows)), along `rows`. No level
/// at or above the table's key cuts them, so one cube's box holds each row group whole, and its
/// writes stop at the group's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedRows {
    rows: Dimension,
}

impl GroupedRows {
    /// Row groups along `rows`.
    pub fn new(rows: Dimension) -> Self {
        Self { rows }
    }

    /// The dimension the rows lie along.
    pub fn rows(self) -> Dimension {
        self.rows
    }

    /// `dimensions` without the rows, which the cubes do not cut.
    pub fn uncut(self, dimensions: &[Dimension]) -> Vec<Dimension> {
        dimensions
            .iter()
            .copied()
            .filter(|&dimension| dimension != self.rows)
            .collect()
    }

    /// Checks the rows are written into a buffer, which masks a group's end.
    ///
    /// # Errors
    ///
    /// [`MATMUL_GROUPED_ROWS_BUFFER_ONLY`], for a sink or an accumulating destination.
    pub fn check_destination(self, destination: MatmulDestination) -> Result<(), Decline> {
        match destination {
            MatmulDestination::Buffer => Ok(()),
            _ => Err(MATMUL_GROUPED_ROWS_BUFFER_ONLY),
        }
    }

    /// Checks `partitioning`, its cubes each holding the levels `inside`, holds a row group in one
    /// cube's box: those levels hold every row of the box, since the stage is sized from the box
    /// while the planes walk what they hold, and the cubes take the boxes in rows.
    ///
    /// # Errors
    ///
    /// [`MATMUL_GROUPED_ROWS_SWIZZLED`] and [`MATMUL_ROW_GROUP_NOT_A_BOX`].
    pub fn check_placed(
        self,
        inside: &Levels,
        partitioning: &MatmulPartitioning,
    ) -> Result<(), Decline> {
        if !matches!(partitioning.cube_order().canonicalize(), CubeOrder::RowMajor) {
            return Err(MATMUL_GROUPED_ROWS_SWIZZLED);
        }
        let held = inside.clone().cubes(&[self.rows]).build()[0]
            .tile(self.rows)
            .expect("a level naming the rows tiles them");
        match held == partitioning.space().extent(self.rows) {
            true => Ok(()),
            false => Err(MATMUL_ROW_GROUP_NOT_A_BOX),
        }
    }
}

/// What a routine whose planes cut the rows declines: grouped rows, which no level cuts.
pub const MATMUL_GROUPED_ROWS_UNREAD: Decline = Decline::Unserved {
    why: "this routine's planes cut the rows, which a table ending row groups places whole",
};

/// What a launch declines whatever the device: a cube box that is not as tall as a row group, which
/// no level cuts.
pub const MATMUL_ROW_GROUP_NOT_A_BOX: Decline = Decline::Unserved {
    why: "a row group is held whole by a box as tall as `max_rows`, which no level cuts",
};

/// What a launch declines whatever the device: grouped rows under a swizzled cube order, which
/// has no cut rows to put back.
pub const MATMUL_GROUPED_ROWS_SWIZZLED: Decline = Decline::Unserved {
    why: "grouped rows are not cut among the cubes, so a swizzled cube order has nothing to permute",
};

/// What a launch declines whatever the device: grouped rows over an lhs stored in tiles, which a
/// row group is not placed over.
pub const MATMUL_GROUPED_ROWS_OVER_TILED_LHS: Decline = Decline::Unserved {
    why: "grouped rows are read in rows, which an lhs stored in tiles is not",
};

/// What a launch declines whatever the device: grouped rows written anywhere but a buffer.
pub const MATMUL_GROUPED_ROWS_BUFFER_ONLY: Decline = Decline::Unserved {
    why: "grouped rows are written into a buffer, which masks their end; a sink or an accumulating destination cannot",
};
