//! The rooflines a round of the few-rows and many-rows autotunes stops at: what a product costs,
//! against the ceiling each autotune's routines are bound by.

use crate::kernels::plan::MatmulCall;
use crate::kernels::tensor::StoredTensor;
use crate::kernels::tune::key::{MatmulBatching, MatmulTuneKey};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::throughput::{MemoryAccess, compute_throughput_key};
use cubecl::tune::Work;
use cubek_std::Instructions;
use cubek_std::autotune::{Ceiling, Roofline};

/// What a product costs, for the roofline a round stops at: the contraction's operations and the
/// bytes of the three operands. A packed input counts its packed bytes, what a kernel streaming it
/// at its stored width reads, so the floor every candidate is measured against is the same. A stack
/// read through routes counts the slabs its batches can reach, one a batch at most.
fn work(key: &MatmulTuneKey, call: &MatmulCall) -> Work {
    let shape = key.shape();
    let bytes = |tensor: &StoredTensor| -> usize {
        tensor
            .buffers()
            .iter()
            .map(|buffer| buffer.size as usize)
            .sum()
    };
    let output = shape.m * shape.n * shape.batch * shape.elem.size();
    let stack = bytes(&call.rhs.tensor);
    let rhs = match shape.rhs.batching {
        MatmulBatching::Slabs { slabs } => stack / slabs * shape.batch.min(slabs),
        MatmulBatching::Shared | MatmulBatching::Batched => stack,
    };
    Work {
        compute_ops: 2 * shape.m * shape.n * shape.k * shape.batch,
        bytes: bytes(&call.lhs.tensor) + rhs + output,
    }
}

/// The few-rows autotune's roofline: the bus, read. A routine that does not stage streams its rhs
/// past a handful of lhs rows and writes a fraction of what it reads, so its ceiling is the read
/// bandwidth at the product's footprint.
pub(super) fn few_rows(key: &MatmulTuneKey, call: &MatmulCall) -> Roofline {
    Roofline {
        client: call.client.clone(),
        ceiling: Ceiling::Memory(MemoryAccess::Read),
        work: work(key, call),
    }
}

/// The many-rows autotune's roofline: both walls, and the key picks. Past the few-rows line the rhs
/// is reused across the rows, so the arithmetic starts to bind, but not on every key a prompt
/// makes: a product's flops per byte is roughly its rows, and the device's own balance sits
/// among them. So the autotune names the largest matrix instruction the device offers at the
/// product's element accumulating in `f32`, plain arithmetic where it offers none, and the read
/// bus, and argues the stop from whichever is higher for the key ([`Ceiling::Either`]).
pub(super) fn many_rows(key: &MatmulTuneKey, call: &MatmulCall) -> Roofline {
    let elem = key.shape().elem;
    let accumulate = ElemType::Float(FloatKind::F32);
    let instruction = call
        .hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .for_triple(elem, elem, accumulate)
        .largest()
        .map(|i| (i.m, i.n, i.k));
    Roofline {
        client: call.client.clone(),
        ceiling: Ceiling::Either {
            compute: compute_throughput_key(instruction, elem, accumulate),
            memory: MemoryAccess::Read,
        },
        work: work(key, call),
    }
}
