//! One gate as its launches read it ([`SwigluCall`]).

use cubecl::client::Client;
use cubek_matmul::kernels::tensor::PlainTensor;

/// One gate, `silu(gate) ⊙ up` over `both`, `(..., 2 · d)` laid out `[gate | up]`, as a launch
/// reads it: the fused rows on `client`'s device.
#[derive(Clone)]
pub struct SwigluCall {
    pub client: Client,
    pub both: PlainTensor,
}
