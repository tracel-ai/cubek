//! What the indexed autotunes, still in this crate until they move to burn-cubecl beside the dense
//! ones, measure with: a candidate's product and the client it lives on ([`MatmulProduct`]), the
//! ceiling a product on matrix instructions is bound by ([`matmul_ceiling`]) and, under the
//! `autotune-checks` feature, how two candidates' outputs are compared ([`assert_equivalent`]).

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::{CmmaDims, MemoryAccess, ThroughputKey};
use cubecl::tune::AutotuneOutput;
use cubek_std::autotune::Ceiling;
use cubek_std::{Hardware, Instructions};

/// What a routed candidate hands back: the product, and the client it lives on, which the
/// autotune checks read two candidates' products through.
#[derive(Clone)]
pub(crate) struct MatmulProduct {
    #[cfg_attr(not(feature = "autotune-checks"), allow(dead_code))]
    pub client: Client,
    pub handle: TensorHandle,
}

impl AutotuneOutput for MatmulProduct {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        assert_equivalent(&self.client, self.handle.clone(), other.handle);
    }
}

/// The ceiling of a product on matrix instructions: the largest the device offers at `elem`
/// accumulating in `f32`, plain arithmetic where it offers none, or the read bus, whichever is
/// higher for the key.
pub(crate) fn matmul_ceiling(hw: &Hardware, elem: ElemType) -> Ceiling {
    let accumulate = ElemType::Float(FloatKind::F32);
    let instruction = hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .for_triple(elem, elem, accumulate)
        .largest()
        .map(|i| CmmaDims {
            m: i.m as usize,
            n: i.n as usize,
            k: i.k as usize,
        });
    Ceiling::Either {
        compute: ThroughputKey::compute(instruction, elem, accumulate),
        memory: MemoryAccess::Read,
    }
}

/// Two candidates' outputs for one problem hold the same floats, within the tolerance two
/// summation orders differ by.
///
/// # Panics
///
/// Where they hold different counts of elements, or differ past the tolerance at one.
#[cfg(feature = "autotune-checks")]
pub(crate) fn assert_equivalent(client: &Client, expected: TensorHandle, actual: TensorHandle) {
    let (expected, actual) = (read_f32(client, expected), read_f32(client, actual));
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

/// `tensor`'s buffer read back as `f32` from whatever float it holds.
#[cfg(feature = "autotune-checks")]
fn read_f32(client: &Client, tensor: TensorHandle) -> Vec<f32> {
    let bytes = client
        .read_one(tensor.handle)
        .expect("the output reads back");
    match tensor.dtype {
        ElemType::Float(FloatKind::F32) => bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
        ElemType::Float(FloatKind::F16) => bytemuck::cast_slice::<u8, half::f16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        ElemType::Float(FloatKind::BF16) => bytemuck::cast_slice::<u8, half::bf16>(&bytes)
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        elem => panic!("an output is a float the routines write, not {elem:?}"),
    }
}
