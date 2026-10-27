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
        let read = |rows: TensorHandle| read_f32(&client, rows);
        let (expected, actual) = (read(self.clone().unpad()), read(other.unpad()));
        assert_eq!(
            expected.len(),
            actual.len(),
            "two candidates for one problem returned different element counts"
        );
        // Relative, with an absolute floor so values near zero do not fail on the last bit of a
        // denormal.
        const RELATIVE: f32 = 1e-2;
        const ABSOLUTE: f32 = 1e-4;
        for (index, (want, got)) in expected.iter().zip(actual.iter()).enumerate() {
            let tolerance = ABSOLUTE + RELATIVE * want.abs();
            assert!(
                (want - got).abs() <= tolerance || (want.is_nan() && got.is_nan()),
                "candidates disagree at element {index}: {want} vs {got}"
            );
        }
    }
}

/// `rows`, row-major and gapless, read back as `f32` from whatever float it holds.
#[cfg(feature = "autotune-checks")]
fn read_f32(client: &cubecl::client::Client, rows: TensorHandle) -> Vec<f32> {
    use cubecl::ir::{ElemType, FloatKind};
    let bytes = client.read_one(rows.handle).expect("the rows read back");
    match rows.dtype {
        ElemType::Float(FloatKind::F32) => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
        ElemType::Float(FloatKind::F16) => bytemuck::cast_slice::<u8, half::f16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        ElemType::Float(FloatKind::BF16) => bytemuck::cast_slice::<u8, half::bf16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        elem => panic!("grouped rows are a float the routines read, not {elem:?}"),
    }
}
