//! What every max pool microkernel computes alone: each output cell the largest tap of its window
//! that lies inside the input.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::shape;
use cubek_microkernel::testing::pool::{PoolLaunch, WindowAxis};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A max pool on the host: its NHWC input, distinct eighths exact in `f16` so every window has
/// one largest tap, and what it writes.
struct HostPool {
    input: Vec<f32>,
    launch: PoolLaunch,
}

impl HostPool {
    fn new(launch: PoolLaunch) -> Self {
        let [h, w] = launch.windowed.input;
        let input = (0..h * w * launch.windowed.channels)
            .map(|i| ((i * 97) % 251) as f32 / 8.0 - 15.0)
            .collect();
        Self { input, launch }
    }

    /// Every output cell's largest tap inside the input, `[1, oh, ow, c]`.
    fn out(&self) -> HostData {
        let launch = &self.launch.windowed;
        let ([h, w], c) = (launch.input, launch.channels);
        let [rows, cols] = launch.windows;
        let [out_h, out_w] = launch.outputs();
        let inside = |window: WindowAxis, output: usize, tap: usize, extent: usize| {
            (output * window.stride + tap * window.dilation)
                .checked_sub(window.padding)
                .filter(|&at| at < extent)
        };
        let mut out = Vec::with_capacity(out_h * out_w * c);
        for oh in 0..out_h {
            for ow in 0..out_w {
                for channel in 0..c {
                    let taps = (0..rows.taps).flat_map(|th| {
                        (0..cols.taps).filter_map(move |tw| {
                            Some((inside(rows, oh, th, h)?, inside(cols, ow, tw, w)?))
                        })
                    });
                    out.push(
                        taps.map(|(ih, iw)| self.input[(ih * w + iw) * c + channel])
                            .fold(f32::NEG_INFINITY, f32::max),
                    );
                }
            }
        }
        HostData::row_major(HostDataVec::F32(out), shape![1, out_h, out_w, c])
    }
}

/// The max pool microkernel `hw` picks for each case.
pub(crate) struct Pool {
    pub hw: Hardware,
}

/// An input of `element` pooled under `windows`, a unit `channel_block` channels a step.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolCase {
    pub element: ElemType,
    pub windows: [WindowAxis; 2],
    pub input: [usize; 2],
    /// Channels per unit of the plane: the case has `channel_block · plane · walks` channels.
    pub channel_block: usize,
    pub walks: usize,
}

impl Pool {
    /// # Errors
    ///
    /// Why the device runs no max pool microkernel at `case`.
    fn launch_at(&self, case: &PoolCase) -> Result<PoolLaunch, Refusal> {
        let channels = case.channel_block * self.hw.dispatch_plane_size() as usize * case.walks;
        PoolLaunch::new(
            &self.hw,
            case.windows,
            case.input,
            channels,
            case.channel_block,
        )
    }
}

impl ScopeUnderTest for Pool {
    type Case = PoolCase;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let launch = self.launch_at(case)?;
        let windowed = launch.windowed;
        let ([h, w], c) = (windowed.input, windowed.channels);
        let [out_h, out_w] = windowed.outputs();
        let host = HostPool::new(launch);
        let input = TestInput::builder(client.clone(), shape![1, h, w, c])
            .dtype(case.element)
            .custom(host.input)
            .generate_without_host_data();
        let out = TestInput::builder(client.clone(), shape![1, out_h, out_w, c])
            .dtype(case.element)
            .custom(vec![-7.0; out_h * out_w * c])
            .generate_without_host_data();
        launch.launch(
            client,
            input.binding().into_tensor_arg(),
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
        vec![HostPool::new(launch).out()]
    }

    /// A max picks one of its taps: exact.
    fn tolerance(&self, _case: &Self::Case) -> f32 {
        0.0
    }
}
