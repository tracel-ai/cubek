//! What every quant test needs: tensors on the device, and the two launches under test.
//!
//! The handles quantize writes are derived from the scheme by
//! [`cubek_test_utils::quant_layout`] rather than spelled out per test: the store decides the
//! values' type and shape, the param decides the scales'. A test that disagrees with the scheme
//! about those is testing its own arithmetic.

use cubecl::{
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
    std::tensor::TensorHandle,
    {TestRuntime, zspace::Shape},
};
use cubek_quant::scheme::QuantScheme;
use cubek_test_utils::{TestInput, quant_layout};

/// `data` on the device as an f32 tensor: what these tests hand the kernels for an input, a block
/// scale grid, or a one-element global scale.
pub(crate) fn f32_tensor(
    client: &ComputeClient<TestRuntime>,
    data: &[f32],
    shape: Shape,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), shape)
        .custom(data.to_vec())
        .generate_without_host_data()
}

/// The block scale grid `scheme` reads over a tensor of `shape`.
pub(crate) fn scale_shape(scheme: &QuantScheme, shape: &Shape) -> Shape {
    let dims: Vec<usize> = shape.iter().copied().collect();

    Shape::from(quant_layout::scales_shape(scheme, &dims))
}

/// The pair quantize writes over a tensor of `shape`. Allocated rather than launched, so a test
/// can hand them to a launch it expects to be refused.
pub(crate) fn quant_outputs(
    client: &ComputeClient<TestRuntime>,
    scheme: &QuantScheme,
    shape: &Shape,
) -> (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>) {
    let dims: Vec<usize> = shape.iter().copied().collect();
    let values = TensorHandle::zeros(
        client,
        Shape::from(quant_layout::values_shape(scheme, &dims)),
        quant_layout::values_dtype(scheme),
    );
    let scales = TensorHandle::zeros(
        client,
        scale_shape(scheme, shape),
        quant_layout::scales_dtype(scheme),
    );

    (values, scales)
}

/// Quantize `input`, calibrated to the block scales in `scale`, into fresh handles.
///
/// The input is always f32 here, which is what [`f32_tensor`] builds.
pub(crate) fn quantize(
    client: &ComputeClient<TestRuntime>,
    scheme: &QuantScheme,
    input: &TensorHandle<TestRuntime>,
    scale: &TensorHandle<TestRuntime>,
    global: Option<&TensorHandle<TestRuntime>>,
    shape: &Shape,
) -> (TensorHandle<TestRuntime>, TensorHandle<TestRuntime>) {
    let (values, scales) = quant_outputs(client, scheme, shape);

    cubek_quant::quantize::launch_ref(
        client,
        input.clone().binding(),
        values.clone().binding(),
        scale.clone().binding(),
        global.map(|global| global.clone().binding()),
        scales.clone().binding(),
        scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    (values, scales)
}

/// Reconstruct `values` into a fresh `out_dtype` tensor, a buffer of its own so an element the
/// kernel never wrote reads as a mismatch rather than as whatever was there before.
pub(crate) fn dequantize(
    client: &ComputeClient<TestRuntime>,
    scheme: &QuantScheme,
    values: &TensorHandle<TestRuntime>,
    scales: &TensorHandle<TestRuntime>,
    global: Option<&TensorHandle<TestRuntime>>,
    shape: &Shape,
    out_dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let out = TensorHandle::zeros(client, shape.clone(), out_dtype);

    cubek_quant::dequantize::launch_ref(
        client,
        values.clone().binding(),
        out.clone().binding(),
        scales.clone().binding(),
        global.map(|global| global.clone().binding()),
        scheme,
        out_dtype,
    )
    .unwrap();

    out
}
