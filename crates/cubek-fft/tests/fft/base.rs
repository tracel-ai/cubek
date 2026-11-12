//! The transforms against the crate's host references: the real forward transform and its
//! inverse, each over a signal or spectrum seeded uniformly or, virtually padded, over one whose
//! buffer holds values past what the kernel may read; and the complex transform there and back.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_fft::eval::cpu_reference::{irfft_ref, rfft_ref};
use cubek_fft::{
    CfftBindings, FftMode, cfft_launch_any_size, irfft_launch, irfft_launch_padded, rfft_launch,
    rfft_launch_padded,
};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, Written};

/// What a refused or failed launch is skipped as, the policy deciding.
const DECLINED: Refusal = Refusal::Declined {
    why: "the transform runs no such launch on this device",
};

/// Transforms no longer than this are held to a tight tolerance: a backend's `sin` and `cos`
/// drift only over long transforms.
pub(crate) const SHORT: usize = 64;

/// Along `dim` of `shape`, the axis transformed.
#[derive(Debug, Clone)]
pub(crate) struct Axis {
    pub shape: Vec<usize>,
    pub dim: usize,
}

impl Axis {
    pub(crate) fn of(shape: &[usize], dim: usize) -> Self {
        Self {
            shape: shape.to_vec(),
            dim,
        }
    }

    /// The same axis, `extent` long.
    fn with_extent(&self, extent: usize) -> Self {
        let mut shape = self.shape.clone();
        shape[self.dim] = extent;
        Self {
            shape,
            dim: self.dim,
        }
    }

    fn len(&self) -> usize {
        self.shape[self.dim]
    }

    /// Uniform values in `[-1, 1)` seeded by `seed` over this shape, in `element`: the tensor, and
    /// its values read back.
    fn seeded(&self, client: &Client, seed: u64, element: ElemType) -> (TensorHandle, HostData) {
        TestInput::builder(client.clone(), self.shape.clone())
            .dtype(element)
            .uniform(seed, -1., 1.)
            .generate_with_f32_host_data()
    }

    /// `values` over this shape, uploaded in `element`.
    fn upload(&self, client: &Client, values: Vec<f32>, element: ElemType) -> TensorHandle {
        TestInput::builder(client.clone(), self.shape.clone())
            .dtype(element)
            .custom(values)
            .generate_without_host_data()
    }

    /// `values` over this shape, on the host.
    fn host(&self, values: Vec<f32>) -> HostData {
        HostData::row_major(HostDataVec::F32(values), Shape::from(self.shape.clone()))
    }
}

/// A smooth wave over every axis of a tensor, so no window repeats another.
#[derive(Debug, Clone, Copy)]
enum Wave {
    Sine,
    Cosine,
}

impl Wave {
    fn at(self, coords: &[usize]) -> f32 {
        let phase: f32 = coords
            .iter()
            .enumerate()
            .map(|(axis, coord)| (axis as f32 + 1.0) * (*coord as f32 + 0.25))
            .sum();
        match self {
            Wave::Sine => phase.sin(),
            Wave::Cosine => phase.cos(),
        }
    }

    /// The wave over `axis`'s shape, zero from `held` on along its axis.
    fn over(self, axis: &Axis, held: usize) -> Vec<f32> {
        let shape = &axis.shape;
        (0..shape.iter().product::<usize>())
            .map(|mut index| {
                let mut coords = vec![0; shape.len()];
                for d in (0..shape.len()).rev() {
                    coords[d] = index % shape[d];
                    index /= shape[d];
                }
                match coords[axis.dim] < held {
                    true => self.at(&coords),
                    false => 0.0,
                }
            })
            .collect()
    }
}

/// The real forward transform along an axis: seeded, or a signal `signal_len` long the kernel
/// zero-pads to `n_fft` itself.
#[derive(Debug, Clone)]
pub(crate) struct Rfft {
    pub axis: Axis,
    pub element: ElemType,
    pub padded: Option<RfftPadding>,
}

/// A signal of `signal_len` samples, transformed as `n_fft`: the buffer holds samples past
/// `signal_len`, which the kernel must not read.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RfftPadding {
    pub signal_len: usize,
    pub n_fft: usize,
}

impl Rfft {
    pub(crate) fn seeded(shape: &[usize], dim: usize) -> Self {
        Self {
            axis: Axis::of(shape, dim),
            element: f32::elem_type_native(),
            padded: None,
        }
    }

    pub(crate) fn in_element(self, element: ElemType) -> Self {
        Self { element, ..self }
    }

    pub(crate) fn padded(shape: &[usize], dim: usize, padding: RfftPadding) -> Self {
        Self {
            padded: Some(padding),
            ..Self::seeded(shape, dim)
        }
    }

    /// The samples the transform covers.
    fn n_fft(&self) -> usize {
        self.padded.map_or(self.axis.len(), |padding| padding.n_fft)
    }

    /// The signal the kernel reads.
    fn signal(&self, client: &Client) -> TensorHandle {
        match self.padded {
            None => self.axis.seeded(client, 42, self.element).0,
            Some(_) => {
                let every = Wave::Sine.over(&self.axis, self.axis.len());
                self.axis.upload(client, every, self.element)
            }
        }
    }

    /// The signal the host transforms: the same, or its first `signal_len` samples zero-padded.
    fn host_signal(&self) -> HostData {
        match self.padded {
            None => {
                let client = cubecl::test_device().client();
                self.axis.seeded(&client, 42, self.element).1
            }
            Some(RfftPadding { signal_len, n_fft }) => {
                let padded = self.axis.with_extent(n_fft);
                padded.host(Wave::Sine.over(&padded, signal_len))
            }
        }
    }
}

/// The real forward transform's launch, against the host's.
pub(crate) struct RfftLaunch;

impl ScopeUnderTest for RfftLaunch {
    type Case = Rfft;

    /// Declines an element the device does no arithmetic in, as Metal `f64`: building the signal
    /// alone reads it back.
    fn launch(&self, client: &Client, case: &Rfft) -> Result<Vec<Written>, Refusal> {
        let usage = client.properties().features.type_usage(case.element);
        if !usage.contains(TypeUsage::Arithmetic) {
            return Err(Refusal::Declined {
                why: "the device does no arithmetic in the element",
            });
        }
        let signal = case.signal(client).binding();
        let spectrum = case.axis.with_extent(case.n_fft() / 2 + 1).shape;
        let re = TensorHandle::zeros(client, spectrum.clone(), case.element);
        let im = TensorHandle::zeros(client, spectrum, case.element);
        let (re_out, im_out) = (re.clone().binding(), im.clone().binding());
        let dim = case.axis.dim;
        let launched = match case.padded {
            Some(padding) => rfft_launch_padded(
                client,
                signal,
                re_out,
                im_out,
                dim,
                padding.signal_len,
                case.element,
            ),
            None => rfft_launch(client, signal, re_out, im_out, dim, case.element),
        };
        launched.map_err(|_| DECLINED)?;
        Ok(vec![
            Written {
                name: "real",
                tensor: re,
                host: HostDataType::F32,
            },
            Written {
                name: "imaginary",
                tensor: im,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, case: &Rfft) -> Vec<HostData> {
        let (re, im) = rfft_ref(&case.host_signal(), case.axis.dim, None);
        vec![re, im]
    }

    /// Tight on a short transform; loose on a long one, whose twiddles a backend's `sin` and
    /// `cos` compute less precisely than the host's, as wgpu's do.
    fn tolerance(&self, case: &Rfft) -> f32 {
        match case.n_fft() <= SHORT {
            true => 1e-3,
            false => 0.4,
        }
    }
}

/// The real inverse transform along an axis: of a seeded spectrum, or of the bins the axis holds,
/// which the kernel zero-pads to the bins of `n_fft` itself.
#[derive(Debug, Clone)]
pub(crate) struct Irfft {
    pub axis: Axis,
    pub padded: Option<IrfftPadding>,
}

/// A spectrum of fewer bins than `n_fft` has, transformed as `n_fft`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IrfftPadding {
    pub n_fft: usize,
}

impl Irfft {
    pub(crate) fn seeded(shape: &[usize], dim: usize) -> Self {
        Self {
            axis: Axis::of(shape, dim),
            padded: None,
        }
    }

    pub(crate) fn padded(shape: &[usize], dim: usize, padding: IrfftPadding) -> Self {
        Self {
            axis: Axis::of(shape, dim),
            padded: Some(padding),
        }
    }

    /// The samples the transform gives back.
    fn n_fft(&self) -> usize {
        self.padded
            .map_or((self.axis.len() - 1) * 2, |padding| padding.n_fft)
    }

    /// The spectrum the kernel reads, real then imaginary: seeded, or two different waves.
    fn spectrum(&self, client: &Client) -> [TensorHandle; 2] {
        let element = f32::elem_type_native();
        match self.padded {
            None => [43, 44].map(|seed| self.axis.seeded(client, seed, element).0),
            Some(_) => [Wave::Sine, Wave::Cosine].map(|wave| {
                let bins = wave.over(&self.axis, self.axis.len());
                self.axis.upload(client, bins, element)
            }),
        }
    }

    /// The spectrum the host transforms: the same, or zero-padded to the bins of `n_fft`.
    fn host_spectrum(&self) -> [HostData; 2] {
        match self.padded {
            None => {
                let client = cubecl::test_device().client();
                [43, 44].map(|seed| self.axis.seeded(&client, seed, f32::elem_type_native()).1)
            }
            Some(IrfftPadding { n_fft }) => {
                let padded = self.axis.with_extent(n_fft / 2 + 1);
                [Wave::Sine, Wave::Cosine]
                    .map(|wave| padded.host(wave.over(&padded, self.axis.len())))
            }
        }
    }
}

/// The real inverse transform's launch, against the host's.
pub(crate) struct IrfftLaunch;

impl ScopeUnderTest for IrfftLaunch {
    type Case = Irfft;

    fn launch(&self, client: &Client, case: &Irfft) -> Result<Vec<Written>, Refusal> {
        let element = f32::elem_type_native();
        let [re, im] = case.spectrum(client);
        let signal =
            TensorHandle::zeros(client, case.axis.with_extent(case.n_fft()).shape, element);
        let (re, im, out) = (re.binding(), im.binding(), signal.clone().binding());
        let (dim, bins) = (case.axis.dim, case.axis.len());
        let launched = match case.padded {
            Some(_) => irfft_launch_padded(client, re, im, out, dim, bins, element),
            None => irfft_launch(client, re, im, out, dim, element),
        };
        launched.map_err(|_| DECLINED)?;
        Ok(vec![Written {
            name: "signal",
            tensor: signal,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Irfft) -> Vec<HostData> {
        let [re, im] = case.host_spectrum();
        vec![irfft_ref(&re, &im, case.axis.dim, None)]
    }

    fn tolerance(&self, case: &Irfft) -> f32 {
        match case.n_fft() <= SHORT {
            true => 1e-3,
            false => 0.01,
        }
    }
}

/// The complex transform forward then inverse along an axis: unnormalized both ways, it gives back
/// the input scaled by the axis's length.
#[derive(Debug, Clone)]
pub(crate) struct CfftRoundTrip {
    pub axis: Axis,
}

impl CfftRoundTrip {
    pub(crate) fn of(shape: &[usize], dim: usize) -> Self {
        Self {
            axis: Axis::of(shape, dim),
        }
    }

    /// The input, real and imaginary, seeded: the tensors, and their values read back.
    fn input(&self, client: &Client) -> [(TensorHandle, HostData); 2] {
        [42, 7].map(|seed| self.axis.seeded(client, seed, f32::elem_type_native()))
    }
}

/// The complex transform's round trip, launched forward then inverse, against the input it
/// scales.
pub(crate) struct CfftRoundTripLaunch;

impl ScopeUnderTest for CfftRoundTripLaunch {
    type Case = CfftRoundTrip;

    fn launch(&self, client: &Client, case: &CfftRoundTrip) -> Result<Vec<Written>, Refusal> {
        let element = f32::elem_type_native();
        let [(input_re, _), (input_im, _)] = case.input(client);
        let shape = case.axis.shape.clone();
        let [spectrum_re, spectrum_im, recovered_re, recovered_im] =
            [(); 4].map(|()| TensorHandle::zeros(client, shape.clone(), element));
        let forward = CfftBindings {
            input_re: input_re.binding(),
            input_im: input_im.binding(),
            output_re: spectrum_re.clone().binding(),
            output_im: spectrum_im.clone().binding(),
        };
        let inverse = CfftBindings {
            input_re: spectrum_re.binding(),
            input_im: spectrum_im.binding(),
            output_re: recovered_re.clone().binding(),
            output_im: recovered_im.clone().binding(),
        };
        let dim = case.axis.dim;
        cfft_launch_any_size(client, forward, dim, element, FftMode::Forward)
            .and_then(|()| cfft_launch_any_size(client, inverse, dim, element, FftMode::Inverse))
            .map_err(|_| DECLINED)?;
        Ok(vec![
            Written {
                name: "real",
                tensor: recovered_re,
                host: HostDataType::F32,
            },
            Written {
                name: "imaginary",
                tensor: recovered_im,
                host: HostDataType::F32,
            },
        ])
    }

    fn reference(&self, case: &CfftRoundTrip) -> Vec<HostData> {
        let n = case.axis.len() as f32;
        case.input(&cubecl::test_device().client())
            .map(|(_, mut host)| {
                if let HostDataVec::F32(values) = &mut host.data {
                    values.iter_mut().for_each(|value| *value *= n);
                }
                host
            })
            .to_vec()
    }

    fn tolerance(&self, _case: &CfftRoundTrip) -> f32 {
        1e-2
    }
}
