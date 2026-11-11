//! What every interpolate microkernel computes alone: each output cell its window's taps inside
//! the input, each weighted, summed.

use crate::launch::interpolate::{Channels, InterpolateLaunch, WindowAxis};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// An interpolation on the host: its NHWC input and its weights, eighths and sixteenths exact in
/// `f16`, and what it writes.
struct HostInterpolate {
    input: Vec<f32>,
    weights: Vec<f32>,
    launch: InterpolateLaunch,
}

impl HostInterpolate {
    fn new(launch: InterpolateLaunch) -> Self {
        let windowed = launch.windowed;
        let [h, w] = windowed.input;
        let [out_h, out_w] = windowed.outputs();
        let [rows, cols] = windowed.windows;
        Self {
            input: (0..h * w * windowed.channels)
                .map(|i| ((i * 97) % 33) as f32 / 8.0 - 2.0)
                .collect(),
            weights: (0..out_h * out_w * rows.taps * cols.taps)
                .map(|i| ((i * 13) % 9) as f32 / 16.0 - 0.25)
                .collect(),
            launch,
        }
    }

    /// Every output cell's weighted sum of its taps inside the input, `[1, oh, ow, c]`.
    fn out(&self) -> HostData {
        let windowed = &self.launch.windowed;
        let ([h, w], c) = (windowed.input, windowed.channels);
        let [rows, cols] = windowed.windows;
        let [out_h, out_w] = windowed.outputs();
        let inside = |window: WindowAxis, output: usize, tap: usize, extent: usize| {
            (output * window.stride + tap * window.dilation)
                .checked_sub(window.padding)
                .filter(|&at| at < extent)
        };
        let mut out = Vec::with_capacity(out_h * out_w * c);
        for oh in 0..out_h {
            for ow in 0..out_w {
                for channel in 0..c {
                    let mut sum = 0.0;
                    for th in 0..rows.taps {
                        for tw in 0..cols.taps {
                            let (Some(ih), Some(iw)) =
                                (inside(rows, oh, th, h), inside(cols, ow, tw, w))
                            else {
                                continue;
                            };
                            let weight =
                                self.weights[((oh * out_w + ow) * rows.taps + th) * cols.taps + tw];
                            sum += weight * self.input[(ih * w + iw) * c + channel];
                        }
                    }
                    out.push(sum);
                }
            }
        }
        HostData::row_major(HostDataVec::F32(out), shape![1, out_h, out_w, c])
    }
}

/// The interpolate microkernel `hw` picks for each case.
pub(crate) struct Interpolate {
    pub hw: Hardware,
}

/// An input of `element` resampled under `windows`, a unit `channel_block` channels a step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterpolateCase {
    pub element: ElemType,
    pub windows: [WindowAxis; 2],
    pub input: [usize; 2],
    /// Channels per unit of the plane: the case has `channel_block · plane · walks` channels.
    pub channel_block: usize,
    pub walks: usize,
}

impl Interpolate {
    /// # Errors
    ///
    /// Why the device runs no interpolate microkernel at `case`.
    fn launch_at(&self, case: &InterpolateCase) -> Result<InterpolateLaunch, Refusal> {
        let per_unit = case.channel_block;
        let count = per_unit * self.hw.dispatch_plane_size() as usize * case.walks;
        InterpolateLaunch::new(
            &self.hw,
            case.windows,
            case.input,
            Channels { count, per_unit },
        )
    }
}

impl ScopeUnderTest for Interpolate {
    type Case = InterpolateCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let windowed = launch.windowed;
        let ([h, w], c) = (windowed.input, windowed.channels);
        let [out_h, out_w] = windowed.outputs();
        let [rows, cols] = windowed.windows;
        let host = HostInterpolate::new(launch);
        let tensor = |values: Vec<f32>, shape| {
            TestInput::builder(client.clone(), shape)
                .dtype(case.element)
                .custom(values)
                .generate_without_host_data()
        };
        let input = tensor(host.input, shape![1, h, w, c]);
        let weights = tensor(host.weights, shape![1, out_h, out_w, rows.taps, cols.taps]);
        let out = tensor(vec![-7.0; out_h * out_w * c], shape![1, out_h, out_w, c]);
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
            weights.binding().into_tensor_arg(),
            out.clone().binding().into_tensor_arg(),
            case.element,
        )?;
        Ok(vec![Written {
            name: "out",
            tensor: out,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        let launch = self
            .launch_at(case)
            .unwrap_or_else(|why| panic!("{case:?} launched: {why:?}"));
        vec![HostInterpolate::new(launch).out()]
    }

    /// Exact in `f32`, the sums of few products of eighths and sixteenths; `f16` rounds the sum.
    fn tolerance(&self, case: &Self::Case) -> f32 {
        match case.element {
            F32 => 1e-6,
            _ => 2e-3,
        }
    }
}
