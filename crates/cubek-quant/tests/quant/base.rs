//! A tensor quantized under a scheme and dequantized back, against the reconstruction the case
//! states: what the format's grid makes of the input, written out by the test rather than derived
//! from the codec under test.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, shape};
use cubek_quant::scheme::QuantScheme;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

use super::harness::{dequantize, f32_tensor, quantize, scale_shape};

/// A type a round trip needs the runtime to hold, beyond `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Needs {
    /// Nothing beyond `f32`.
    Nothing,
    /// An 8-bit type, signed and unsigned, for an fp8 value stored natively or a `ue8m0` scale:
    /// WGSL has none, and packs fp8 four values to a `u32`.
    Byte,
}

/// One round trip: `input` of `shape` under `scheme`, calibrated to `scales` (and to `global`, for
/// a two-level scheme), written back in `out`, which must reconstruct as `expected`.
#[derive(Debug, Clone)]
pub(crate) struct RoundTrip {
    pub scheme: QuantScheme,
    pub shape: Shape,
    pub input: Vec<f32>,
    pub scales: Vec<f32>,
    pub global: Option<f32>,
    pub out: ElemType,
    pub expected: Vec<f32>,
    pub tolerance: f32,
    pub needs: Needs,
}

impl RoundTrip {
    /// `input`, one row of it, at a scale of one under `scheme`, which must come back as
    /// `expected` exactly, in `f32`: the codec's answer and nothing else.
    pub(crate) fn at_unit_scale(scheme: QuantScheme, input: &[f32], expected: &[f32]) -> Self {
        Self {
            global: scheme
                .tensor_scale()
                .filter(|_| scheme.block_scale().is_some())
                .map(|_| 1.0),
            scheme,
            shape: shape![1, input.len()],
            input: input.to_vec(),
            scales: vec![1.0],
            out: ElemType::Float(FloatKind::F32),
            expected: expected.to_vec(),
            tolerance: 0.,
            needs: Needs::Nothing,
        }
    }

    /// The same round trip, its single block calibrated to `scale`.
    pub(crate) fn scaled(self, scale: f32) -> Self {
        Self {
            scales: vec![scale],
            ..self
        }
    }

    /// The same round trip, run only where the runtime holds what `needs` names.
    pub(crate) fn needing(self, needs: Needs) -> Self {
        Self { needs, ..self }
    }

    /// Whether the runtime holds what the round trip needs.
    fn held(&self, client: &Client) -> bool {
        match self.needs {
            Needs::Nothing => true,
            Needs::Byte => {
                i8::supported_uses(client).contains(TypeUsage::Conversion)
                    && u8::supported_uses(client).contains(TypeUsage::Conversion)
            }
        }
    }
}

/// Quantize, then dequantize, against the reconstruction the case states.
pub(crate) struct Quantized;

impl ScopeUnderTest for Quantized {
    type Case = RoundTrip;

    fn launch(&self, client: &Client, trip: &RoundTrip) -> Result<Vec<Written>, Refusal> {
        if !trip.held(client) {
            return Err(Refusal::Declined {
                why: "the runtime holds no type the round trip needs",
            });
        }
        let input = f32_tensor(client, &trip.input, trip.shape.clone());
        let scales = f32_tensor(client, &trip.scales, scale_shape(&trip.scheme, &trip.shape));
        let global = trip
            .global
            .map(|global| f32_tensor(client, &[global], shape![1]));
        let (values, stored, stored_global) = quantize(
            client,
            &trip.scheme,
            &input,
            &scales,
            global.as_ref(),
            &trip.shape,
        );
        let restored = dequantize(
            client,
            &trip.scheme,
            &values,
            &stored,
            stored_global.as_ref(),
            &trip.shape,
            trip.out,
        );
        Ok(vec![Written {
            name: "restored",
            tensor: restored,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, trip: &RoundTrip) -> Vec<HostData> {
        vec![HostData::row_major(
            HostDataVec::F32(trip.expected.clone()),
            trip.shape.clone(),
        )]
    }

    fn tolerance(&self, trip: &RoundTrip) -> f32 {
        trip.tolerance
    }
}
