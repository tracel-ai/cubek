use cubecl::std::tensor::TensorHandle;

use super::GroupedRows;
use crate::kernels::tensor::PlainTensor;

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
    /// # Panics
    ///
    /// On rows other than the padded rows `layout` lays out.
    pub fn new(rows: TensorHandle, layout: GroupedRows) -> Self {
        assert_eq!(
            rows.shape()[0],
            layout.padded_rows(),
            "the padded rows this layout lays out"
        );
        Self { rows, layout }
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
    /// [`GroupedRows::combine_routes`].
    pub fn combine_routes(
        &self,
        routes: PlainTensor,
        place_of_route: PlainTensor,
        weights: PlainTensor,
    ) -> TensorHandle {
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
