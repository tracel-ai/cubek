//! The interleaved transforms against a direct DFT on the host. Each output is read back as its
//! whole buffer, gaps and margins included, so a value written where the layout puts none fails
//! as surely as a wrong one.
//!
//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0).

use core::f32::consts::PI;

use cubecl::CubeElement;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_fft::{
    ComplexTensorHandle, FftMode, FftNormalization, cfft_interleaved_launch,
    irfft_interleaved_launch_padded, rfft_interleaved_launch_padded,
};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};
use num_complex::Complex32;

use crate::fft::SHORT;

/// What a buffer holds where its layout puts no value: an output must keep it there, and an input
/// read there spoils the transform.
const UNPLACED: f32 = 1e4;

/// Every normalization a transform takes.
const NORMALIZATIONS: [FftNormalization; 3] = [
    FftNormalization::None,
    FftNormalization::ByN,
    FftNormalization::Ortho,
];

/// The longest complex transform one cube's shared memory holds in `f32`: the last the shared
/// path takes, twice it the first the four-step split takes.
pub(crate) fn longest_shared(client: &Client) -> usize {
    let elements = client.properties().hardware.max_shared_memory_size / (2 * size_of::<f32>());
    match elements.is_power_of_two() {
        true => elements,
        false => elements.next_power_of_two() / 2,
    }
}

/// Where a tensor's values lie in its buffer: its shape and strides, in its own elements, between
/// `margin` scalars at each end that its binding leaves out.
#[derive(Debug, Clone)]
pub(crate) struct Placement {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub margin: usize,
}

impl Placement {
    pub(crate) fn contiguous(shape: &[usize]) -> Self {
        let mut strides = vec![1; shape.len()];
        for axis in (1..shape.len()).rev() {
            strides[axis - 1] = strides[axis] * shape[axis];
        }
        Self::strided(shape, &strides)
    }

    pub(crate) fn strided(shape: &[usize], strides: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            margin: 0,
        }
    }

    /// The same layout, bound `margin` scalars into its buffer and as many from its end.
    pub(crate) fn with_margin(self, margin: usize) -> Self {
        Self { margin, ..self }
    }

    fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// The buffer's scalars, `width` an element: both margins, and every scalar from the first
    /// element's to the last's.
    fn scalars(&self, width: usize) -> usize {
        let last: usize = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(extent, stride)| (extent - 1) * stride)
            .sum();
        2 * self.margin + (last + 1) * width
    }

    /// Where each element lies, in elements, in row-major order.
    fn offsets(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|mut index| {
            let mut offset = 0;
            for axis in (0..self.shape.len()).rev() {
                offset += index % self.shape[axis] * self.strides[axis];
                index /= self.shape[axis];
            }
            offset
        })
    }

    /// The row-major index of each element along `dim`, one lane for each index of the other axes,
    /// in row-major order: two layouts differing only along `dim` list their lanes alike.
    fn lanes(&self, dim: usize) -> Vec<Vec<usize>> {
        let extent = self.shape[dim];
        let inner: usize = self.shape[dim + 1..].iter().product();
        let outer: usize = self.shape[..dim].iter().product();
        (0..outer)
            .flat_map(|outer| {
                (0..inner).map(move |inner_index| {
                    (0..extent)
                        .map(|position| (outer * extent + position) * inner + inner_index)
                        .collect()
                })
            })
            .collect()
    }

    /// `values`, `width` scalars an element in row-major order, where this layout puts them, and
    /// [`UNPLACED`] everywhere else.
    fn buffer(&self, values: &[f32], width: usize) -> Vec<f32> {
        let mut buffer = vec![UNPLACED; self.scalars(width)];
        for (index, offset) in self.offsets().enumerate() {
            let at = self.margin + offset * width;
            buffer[at..at + width].copy_from_slice(&values[index * width..(index + 1) * width]);
        }
        buffer
    }

    /// [`buffer`](Self::buffer) on the device, its binding without the margins.
    fn upload(&self, client: &Client, values: &[f32], width: usize) -> Handle {
        let margin = (self.margin * size_of::<f32>()) as u64;
        client
            .create_from_slice(f32::as_bytes(&self.buffer(values, width)))
            .offset_start(margin)
            .offset_end(margin)
    }

    /// Complex `values`, real and imaginary scalars in turn, placed on the device.
    pub(crate) fn complex(&self, client: &Client, values: &[f32]) -> ComplexTensorHandle {
        ComplexTensorHandle::new_strided(
            self.shape.clone(),
            self.strides.clone(),
            self.upload(client, values, 2),
            f32::elem_type_native(),
        )
        .unwrap_or_else(|why| panic!("{self:?} holds a complex tensor: {why}"))
    }

    /// Real `values` placed on the device.
    pub(crate) fn real(&self, client: &Client, values: &[f32]) -> TensorHandle {
        TensorHandle::new(
            self.upload(client, values, 1),
            self.shape.clone(),
            self.strides.clone(),
            f32::elem_type_native(),
        )
    }

    /// `tensor`'s whole buffer, margins included, as the transform left it.
    fn written(&self, name: &'static str, tensor: TensorHandle, width: usize) -> Written {
        let mut handle = tensor.handle;
        handle.offset_start = None;
        handle.offset_end = None;
        Written {
            name,
            tensor: TensorHandle::new(
                handle,
                vec![self.scalars(width)],
                vec![1],
                f32::elem_type_native(),
            ),
            host: HostDataType::F32,
        }
    }

    /// The whole buffer holding `values` as [`written`](Self::written) reads one.
    fn expected(&self, values: &[f32], width: usize) -> HostData {
        HostData::row_major(
            HostDataVec::F32(self.buffer(values, width)),
            Shape::from(vec![self.scalars(width)]),
        )
    }

    /// An output's buffer before the transform: [`UNPLACED`] throughout.
    fn unwritten(&self, width: usize) -> Vec<f32> {
        vec![UNPLACED; self.len() * width]
    }
}

/// The values a transform reads: every element distinct and complex, or only the first two
/// along the transformed axis, which keeps a long transform's direct DFT short.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Values {
    Ramp,
    Pair,
}

impl Values {
    /// The values over `shape`, in row-major order, transformed along `dim`.
    fn over(self, shape: &[usize], dim: usize) -> Vec<Complex32> {
        let inner: usize = shape[dim + 1..].iter().product();
        (0..shape.iter().product::<usize>())
            .map(|index| match (self, index / inner % shape[dim]) {
                (Values::Ramp, _) => {
                    let index = index as f32;
                    Complex32::new(index * 0.5 - 1.0, 0.25 - index * 0.75)
                }
                (Values::Pair, 0) => Complex32::new(0.75, -1.25),
                (Values::Pair, 1) => Complex32::new(-0.5, 2.0),
                (Values::Pair, _) => Complex32::ZERO,
            })
            .collect()
    }
}

/// The twiddles of an `n`-point DFT turning the way `sign` says: `-1` forward, `1` inverse.
struct Twiddles {
    sign: f32,
    n: usize,
}

impl Twiddles {
    /// `exp(sign · 2πi · k · j / n)`, its angle reduced first so a long transform loses nothing
    /// to it.
    fn at(&self, k: usize, j: usize) -> Complex32 {
        let turn = ((k * j) % self.n) as f32 / self.n as f32;
        Complex32::from_polar(1.0, self.sign * 2.0 * PI * turn)
    }
}

/// The complex transform of `values` along `dim`, its output laid out as its input.
#[derive(Debug, Clone)]
pub(crate) struct InterleavedCfft {
    pub placement: Placement,
    pub dim: usize,
    pub values: Values,
    pub mode: FftMode,
    pub normalization: FftNormalization,
}

impl InterleavedCfft {
    /// Forward and inverse, under every normalization.
    pub(crate) fn every_way(
        placement: Placement,
        dim: usize,
        values: Values,
    ) -> impl Iterator<Item = Self> {
        [FftMode::Forward, FftMode::Inverse]
            .into_iter()
            .flat_map(move |mode| {
                let placement = placement.clone();
                NORMALIZATIONS.map(move |normalization| Self {
                    placement: placement.clone(),
                    dim,
                    values,
                    mode,
                    normalization,
                })
            })
    }

    fn input(&self) -> Vec<Complex32> {
        self.values.over(&self.placement.shape, self.dim)
    }

    fn n_fft(&self) -> usize {
        self.placement.shape[self.dim]
    }
}

/// The interleaved complex transform, against a direct DFT.
pub(crate) struct CfftInterleavedLaunch;

impl ScopeUnderTest for CfftInterleavedLaunch {
    type Case = InterleavedCfft;

    fn launch(&self, client: &Client, case: &InterleavedCfft) -> Result<Vec<Written>, Refusal> {
        let placement = &case.placement;
        let input: Vec<f32> = case.input().iter().flat_map(|x| [x.re, x.im]).collect();
        let input = placement.complex(client, &input);
        let output = placement.complex(client, &placement.unwritten(2));
        cfft_interleaved_launch(
            client,
            input.binding(),
            output.binding(),
            case.dim,
            case.mode,
            case.normalization,
        )
        .unwrap_or_else(|why| panic!("{case:?} launched: {why}"));
        Ok(vec![placement.written(
            "output",
            output.into_raw_parts(),
            2,
        )])
    }

    /// `Σ_j x[j] · exp(∓2πi · k · j / n)` over each lane, scaled as the normalization says; a zero
    /// sample adds nothing, so it is left out.
    fn reference(&self, case: &InterleavedCfft) -> Vec<HostData> {
        let input = case.input();
        let n = case.n_fft();
        let twiddles = Twiddles {
            sign: match case.mode {
                FftMode::Forward => -1.0,
                FftMode::Inverse => 1.0,
            },
            n,
        };
        let scale = case.normalization.scale_f32(n).unwrap();
        let mut output = vec![Complex32::ZERO; input.len()];
        for lane in case.placement.lanes(case.dim) {
            let samples: Vec<(usize, Complex32)> = lane
                .iter()
                .map(|&index| input[index])
                .enumerate()
                .filter(|(_, x)| *x != Complex32::ZERO)
                .collect();
            for (k, &index) in lane.iter().enumerate() {
                output[index] = scale
                    * samples
                        .iter()
                        .map(|&(j, x)| x * twiddles.at(k, j))
                        .sum::<Complex32>();
            }
        }
        let output: Vec<f32> = output.iter().flat_map(|x| [x.re, x.im]).collect();
        vec![case.placement.expected(&output, 2)]
    }

    /// Tight on a short transform; past it, a backend's `sin` and `cos` drift from the host's.
    fn tolerance(&self, case: &InterleavedCfft) -> f32 {
        match case.n_fft() <= SHORT {
            true => 1e-4,
            false => 2e-3,
        }
    }
}

/// The real forward transform of a signal's first `signal_len` samples along `dim`, zero-padded to
/// the `n_fft` its spectrum's `n_fft / 2 + 1` bins say.
#[derive(Debug, Clone)]
pub(crate) struct InterleavedRfft {
    pub signal: Placement,
    pub spectrum: Placement,
    pub dim: usize,
    pub signal_len: usize,
    pub values: Values,
    pub normalization: FftNormalization,
}

impl InterleavedRfft {
    /// The transform under every normalization.
    pub(crate) fn every_normalization(
        signal: Placement,
        spectrum: Placement,
        dim: usize,
        signal_len: usize,
        values: Values,
    ) -> impl Iterator<Item = Self> {
        NORMALIZATIONS
            .map(move |normalization| Self {
                signal: signal.clone(),
                spectrum: spectrum.clone(),
                dim,
                signal_len,
                values,
                normalization,
            })
            .into_iter()
    }

    /// The signal: the real part of `values`.
    fn samples(&self) -> Vec<f32> {
        let values = self.values.over(&self.signal.shape, self.dim);
        values.iter().map(|x| x.re).collect()
    }

    fn n_fft(&self) -> usize {
        2 * (self.spectrum.shape[self.dim] - 1)
    }
}

/// The interleaved real forward transform, against a direct DFT.
pub(crate) struct RfftInterleavedLaunch;

impl ScopeUnderTest for RfftInterleavedLaunch {
    type Case = InterleavedRfft;

    fn launch(&self, client: &Client, case: &InterleavedRfft) -> Result<Vec<Written>, Refusal> {
        let signal = case.signal.real(client, &case.samples());
        let spectrum = case.spectrum.complex(client, &case.spectrum.unwritten(2));
        rfft_interleaved_launch_padded(
            client,
            &signal,
            spectrum.binding(),
            case.dim,
            case.signal_len,
            case.normalization,
        )
        .unwrap_or_else(|why| panic!("{case:?} launched: {why}"));
        Ok(vec![case.spectrum.written(
            "spectrum",
            spectrum.into_raw_parts(),
            2,
        )])
    }

    /// `X[k] = Σ_{t < signal_len} x[t] · exp(-2πi · k · t / n_fft)` over each lane, scaled as the
    /// normalization says.
    fn reference(&self, case: &InterleavedRfft) -> Vec<HostData> {
        let samples = case.samples();
        let n = case.n_fft();
        let scale = case.normalization.scale_f32(n).unwrap();
        let twiddles = Twiddles { sign: -1.0, n };
        let mut spectrum = vec![Complex32::ZERO; case.spectrum.len()];
        let lanes = case.signal.lanes(case.dim);
        for (signal, bins) in lanes.iter().zip(case.spectrum.lanes(case.dim)) {
            for (k, &index) in bins.iter().enumerate() {
                spectrum[index] = scale
                    * signal[..case.signal_len]
                        .iter()
                        .enumerate()
                        .filter(|&(_, &at)| samples[at] != 0.0)
                        .map(|(t, &at)| samples[at] * twiddles.at(k, t))
                        .sum::<Complex32>();
            }
        }
        let spectrum: Vec<f32> = spectrum.iter().flat_map(|x| [x.re, x.im]).collect();
        vec![case.spectrum.expected(&spectrum, 2)]
    }

    /// Tight on a short transform; past it, a backend's `sin` and `cos` drift from the host's.
    fn tolerance(&self, case: &InterleavedRfft) -> f32 {
        match case.n_fft() <= SHORT {
            true => 1e-4,
            false => 2e-3,
        }
    }
}

/// The real inverse transform along `dim` of a spectrum's first `bins` bins, zero-padded to the
/// `n_fft / 2 + 1` of the `n_fft` samples its signal holds.
#[derive(Debug, Clone)]
pub(crate) struct InterleavedIrfft {
    pub spectrum: Placement,
    pub signal: Placement,
    pub dim: usize,
    pub bins: usize,
    pub values: Values,
    pub normalization: FftNormalization,
}

impl InterleavedIrfft {
    /// The transform under every normalization.
    pub(crate) fn every_normalization(
        spectrum: Placement,
        signal: Placement,
        dim: usize,
        bins: usize,
        values: Values,
    ) -> impl Iterator<Item = Self> {
        NORMALIZATIONS
            .map(move |normalization| Self {
                spectrum: spectrum.clone(),
                signal: signal.clone(),
                dim,
                bins,
                values,
                normalization,
            })
            .into_iter()
    }

    /// The spectrum: `values`, the first bin and the middle one real, as a real signal's are.
    fn spectrum(&self) -> Vec<Complex32> {
        let mut spectrum = self.values.over(&self.spectrum.shape, self.dim);
        for lane in self.spectrum.lanes(self.dim) {
            for k in [0, self.n_fft() / 2] {
                if let Some(&index) = lane.get(k) {
                    spectrum[index].im = 0.0;
                }
            }
        }
        spectrum
    }

    fn n_fft(&self) -> usize {
        self.signal.shape[self.dim]
    }
}

/// The interleaved real inverse transform, against a direct inverse DFT.
pub(crate) struct IrfftInterleavedLaunch;

impl ScopeUnderTest for IrfftInterleavedLaunch {
    type Case = InterleavedIrfft;

    fn launch(&self, client: &Client, case: &InterleavedIrfft) -> Result<Vec<Written>, Refusal> {
        let bins: Vec<f32> = case.spectrum().iter().flat_map(|x| [x.re, x.im]).collect();
        let spectrum = case.spectrum.complex(client, &bins);
        let signal = case.signal.real(client, &case.signal.unwritten(1));
        irfft_interleaved_launch_padded(
            client,
            spectrum.binding(),
            &signal,
            case.dim,
            case.bins,
            case.normalization,
        )
        .unwrap_or_else(|why| panic!("{case:?} launched: {why}"));
        Ok(vec![case.signal.written("signal", signal, 1)])
    }

    /// `x[t] = Σ_{k < bins} c_k · Re(X[k] · exp(2πi · k · t / n_fft))` over each lane, `c_k` one
    /// for the first bin and the middle one and two for every other, which stands for its mirror;
    /// scaled as the normalization says.
    fn reference(&self, case: &InterleavedIrfft) -> Vec<HostData> {
        let spectrum = case.spectrum();
        let n = case.n_fft();
        let scale = case.normalization.scale_f32(n).unwrap();
        let twiddles = Twiddles { sign: 1.0, n };
        let mut signal = vec![0.0; case.signal.len()];
        let lanes = case.spectrum.lanes(case.dim);
        for (bins, samples) in lanes.iter().zip(case.signal.lanes(case.dim)) {
            for (t, &index) in samples.iter().enumerate() {
                signal[index] = scale
                    * bins[..case.bins]
                        .iter()
                        .enumerate()
                        .map(|(k, &at)| {
                            let mirrored = match k == 0 || 2 * k == n {
                                true => 1.0,
                                false => 2.0,
                            };
                            mirrored * (spectrum[at] * twiddles.at(k, t)).re
                        })
                        .sum::<f32>();
            }
        }
        vec![case.signal.expected(&signal, 1)]
    }

    /// Tight on a short transform; past it, a backend's `sin` and `cos` drift from the host's.
    fn tolerance(&self, case: &InterleavedIrfft) -> f32 {
        match case.n_fft() <= SHORT {
            true => 1e-4,
            false => 2e-3,
        }
    }
}
