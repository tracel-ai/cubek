//! One norm as its launches read it ([`RmsNormCall`]).

use cubecl::client::Client;
use cubek_matmul::kernels::tensor::PlainTensor;

/// One norm, `input / sqrt(mean(input², last) + epsilon) · weight`, as a launch reads it: the rows
/// and the weight on `client`'s device.
#[derive(Clone)]
pub struct RmsNormCall {
    pub client: Client,
    pub input: PlainTensor,
    pub weight: PlainTensor,
    pub epsilon: f32,
}
