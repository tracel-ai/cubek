//! A mean of integers divides in the integers its values accumulate in, which hold totals no `f32`
//! does. The `reduce_dim` matrix reduces values an `f32` holds.

use cubecl::{prelude::*, std::tensor::TensorHandle};
use cubek_reduce::ReduceOperation;
use cubek_reduce::kernels::{Reduce, ReduceElements};

use crate::reduce::it::test_case::UNIT;

#[test]
fn a_mean_of_i32_keeps_the_digits_an_f32_drops() {
    /// One past the integers an `f32` holds one by one.
    const VALUE: i32 = (1 << 24) + 1;
    const ROW_VALUE_COUNT: usize = 5;
    let client = cubecl::test_device().client();
    let i32_dtype = i32::elem_type_native();

    let row = client.create_from_slice(i32::as_bytes(&[VALUE; ROW_VALUE_COUNT]));
    let input = TensorHandle::new_contiguous([1, ROW_VALUE_COUNT], row, i32_dtype);
    let out = TensorHandle::empty(&client, [1, 1], i32_dtype);

    Reduce::new(UNIT)
        .execute(
            &client,
            input.binding(),
            out.clone().binding(),
            1,
            ReduceOperation::Mean,
            ReduceElements::for_operation(ReduceOperation::Mean, i32_dtype, None),
        )
        .expect("the unit kernel should serve every problem");

    let written = client.read_one_unchecked(out.handle);
    assert_eq!(i32::from_bytes(&written)[0], VALUE);
}
