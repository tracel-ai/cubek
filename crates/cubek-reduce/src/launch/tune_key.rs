use cubecl::{AutotuneKey, ir::ElemType};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct ReduceAutotuneKey {
    elem_input: ElemType,
    elem_output: ElemType,
    elem_acc: ElemType,
    /// Whether the axis is contiguous.
    pub axis_is_contiguous: bool,
    /// The length of the vector to reduce.
    ///
    /// # Notes
    ///
    /// Max is 4^4, so 5 values are possible.
    #[autotune(anchor(exp(min = 16, max = 1024, base = 4)))]
    pub vector_size: usize,
    /// The number of vectors to reduce.
    ///
    /// # Notes
    ///
    /// Max is 8^5, so 5 values are possible.
    #[autotune(anchor(exp(max = 32768, base = 8)))]
    pub vector_count: usize,
}

impl ReduceAutotuneKey {
    pub fn generate(
        elem_input: ElemType,
        elem_output: ElemType,
        elem_acc: ElemType,
        input_shape: &[usize],
        axis_is_contiguous: bool,
        axis: usize,
    ) -> Self {
        let rank = input_shape.len();

        if axis > rank {
            panic!("axis {axis} is out-of-bound for a rank of {rank}");
        }

        let reduce_axis_shape = input_shape[axis];

        let reduce_count = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, shape)| (i != axis).then_some(shape))
            .product();

        ReduceAutotuneKey::new(
            elem_input,
            elem_output,
            elem_acc,
            axis_is_contiguous,
            reduce_axis_shape,
            reduce_count,
        )
    }
}
