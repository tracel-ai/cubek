use cubecl::zspace::shape;
use cubek_test_utils::{StridedLayout, TestInput};

#[test]
fn host_data_typed_accessors_and_iter() {
    let client = cubecl::test_device().client();

    // arange tensor: values 0..6 in row-major order.
    let host = TestInput::builder(client.clone(), shape![2, 3])
        .arange()
        .f32_host_data();

    // Panicking and falling-back accessors agree on F32 data.
    assert_eq!(host.get_f32(&[1, 2]), 5.0);
    assert_eq!(host.try_get_f32(&[1, 2]), Some(5.0));
    // Wrong dtype → None instead of panic.
    assert_eq!(host.try_get_i32(&[1, 2]), None);
    assert_eq!(host.try_get_bool(&[1, 2]), None);

    // Indexed iterator walks every cell row-major.
    let collected: Vec<(Vec<usize>, f32)> = host.iter_indexed_f32().collect();
    assert_eq!(collected.len(), 6);
    assert_eq!(collected[0], (vec![0, 0], 0.0));
    assert_eq!(collected[3], (vec![1, 0], 3.0));
    assert_eq!(collected.last().unwrap(), &(vec![1, 2], 5.0));
}

#[test]
fn host_data_iter_respects_strides() {
    let client = cubecl::test_device().client();

    // Same logical 2x3 tensor, but stored col-major. The iterator must walk
    // the *logical* row-major order while resolving each cell through the
    // strides, so the values must match a row-major arange.
    let actual = TestInput::builder(client.clone(), shape![2, 3])
        .layout(StridedLayout::ColMajor)
        .arange()
        .f32_host_data();

    let collected: Vec<f32> = actual.iter_indexed_f32().map(|(_, v)| v).collect();
    assert_eq!(collected, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}
