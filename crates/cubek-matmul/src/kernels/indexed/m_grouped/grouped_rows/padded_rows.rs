#[cfg(doc)]
use super::COMBINED_ROUTES_UNREAD;
use super::{GATED_ROWS_UNREAD, GROUPED_ROWS_UNREAD, GroupedRows};
use crate::kernels::tensor::PlainTensor;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;

/// Rows in the padded layout, beside the layout that places them: what a grouped product hands
/// back, for the next step to read padded or to [`unpad`](Self::unpad).
#[derive(Clone)]
pub struct PaddedRows {
    /// `[tiles · align, width]`, the padding rows holding nothing a reader may use.
    rows: TensorHandle,
    layout: GroupedRows,
}

impl PaddedRows {
    /// `rows`, `[tiles · align, width]`, as `layout` places them.
    ///
    /// # Errors
    ///
    /// [`GROUPED_ROWS_UNREAD`] for rows other than the padded rows `layout` lays out.
    pub fn new(rows: TensorHandle, layout: GroupedRows) -> Result<Self, Decline> {
        let shape = rows.shape();
        if shape.rank() != 2 || shape[0] != layout.padded_rows() {
            return Err(GROUPED_ROWS_UNREAD);
        }
        Ok(Self { rows, layout })
    }

    /// The same rows as `layout` places them.
    ///
    /// # Assumptions
    ///
    /// `layout` lays out as many padded rows as the one that places them now.
    pub(crate) fn placed_by(&self, layout: GroupedRows) -> Self {
        Self {
            rows: self.rows.clone(),
            layout,
        }
    }

    /// Whether the rows are a `[gate | up]` output over `groups` groups, which another layout of
    /// those groups gates.
    ///
    /// # Errors
    ///
    /// [`GATED_ROWS_UNREAD`].
    pub(crate) fn admit_gated(&self, groups: usize) -> Result<(), Decline> {
        if self.layout.groups() != groups || !self.rows.shape()[1].is_multiple_of(2) {
            return Err(GATED_ROWS_UNREAD);
        }
        Ok(())
    }

    /// The padded rows, `[tiles · align, width]`.
    pub fn rows(&self) -> &TensorHandle {
        &self.rows
    }

    /// The layout that places the rows.
    pub fn layout(&self) -> &GroupedRows {
        &self.layout
    }

    /// The padded rows and their layout, for a launch that binds the rows and reads the layout.
    pub fn into_parts(self) -> (TensorHandle, GroupedRows) {
        (self.rows, self.layout)
    }

    /// The rows as they were grouped, padding dropped.
    pub fn unpad(self) -> TensorHandle {
        self.layout.unpad(PlainTensor::of_handle(self.rows))
    }

    /// Every token's routes read back out of these rows and summed under their weights: see
    /// `GroupedRows::combine_routes`.
    ///
    /// # Assumptions
    ///
    /// Every route that names a group names the one its place lies in.
    ///
    /// # Errors
    ///
    /// [`COMBINED_ROUTES_UNREAD`].
    pub fn combine_routes(
        &self,
        routes: PlainTensor,
        place_of_route: PlainTensor,
        weights: PlainTensor,
    ) -> Result<TensorHandle, Decline> {
        self.layout
            .combine_routes(&self.rows, routes, place_of_route, weights)
    }
}

/// What lets padded rows be an autotune candidate's output. Two candidates may pad to different
/// tiles, so they are compared as the rows they place.
impl cubecl::tune::AutotuneOutput for PaddedRows {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        let client = self.layout.client().clone();
        crate::kernels::tune::assert_equivalent(&client, self.clone().unpad(), other.unpad());
    }
}
