//! The transforms against the crate's host references: the real forward transform and its
//! inverse, each over a signal or spectrum seeded uniformly or, virtually padded, over one whose
//! padding the kernel never reads; and the complex transform there and back.

use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
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

    /// `shape` with `dim` `extent` long.
    fn with_extent(&self, extent: usize) -> Vec<usize> {
        let mut shape = self.shape.clone();
        shape[self.dim] = extent;
        shape
    }

    /// The value at `coords`: a smooth wave over every axis, so no window repeats another.
    fn sample(coords: &[usize]) -> f32 {
        coords
            .iter()
            .enumerate()
            .map(|(axis, coord)| (axis as f32 + 1.0) * (*coord as f32 + 0.25))
            .sum::<f32>()
            .sin()
    }

    /// The samples of a tensor of `shape`, zero past `len` along `dim`.
    fn samples(shape: &[usize], dim: usize, len: usize) -> Vec<f32> {
        (0..shape.iter().product::<usize>())
            .map(|mut index| {
                let mut coords = vec![0; shape.len()];
                for axis in (0..shape.len()).rev() {
                    coords[axis] = index % shape[axis];
                    index /= shape[axis];
                }
                match coords[dim] < len {
                    true => Self::sample(&coords),
                    false => 0.0,
                }
            })
            .collect()
    }

    /// `values` of `shape` in `element`, its tensor and its values on the host.
    fn tensor(
        client: &Client,
        shape: Vec<usize>,
        values: Vec<f32>,
        element: ElemType,
    ) -> (TensorHandle, HostData) {
        TestInput::builder(client.clone(), shape)
            .dtype(element)
            .custom(values)
            .generate_with_f32_host_data()
    }

    /// Uniform values in `[-1, 1)` seeded by `seed`, of `shape` in `element`.
    fn seeded(
        client: &Client,
        shape: Vec<usize>,
        seed: u64,
        element: ElemType,
    ) -> (TensorHandle, HostData) {
        TestInput::builder(client.clone(), shape)
            .dtype(element)
            .uniform(seed, -1., 1.)
            .generate_with_f32_host_data()
    }

    fn zeros(client: &Client, shape: Vec<usize>, element: ElemType) -> TensorHandle {
        TestInput::builder(client.clone(), shape)
            .dtype(element)
            .zeros()
            .generate_without_host_data()
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

/// A signal of `signal_len` samples, transformed as `n_fft`: what lies past `signal_len` in the
/// buffer is never read.
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

    /// The signal the kernel reads, and the one the host transforms: the same, or zero-padded.
    fn signal(&self, client: &Client) -> (TensorHandle, HostData) {
        let Axis { shape, dim } = &self.axis;
        match self.padded {
            None => Axis::seeded(client, shape.clone(), 42, self.element),
            Some(RfftPadding { signal_len, n_fft }) => {
                let read = Axis::samples(shape, *dim, signal_len);
                let (tensor, _) = Axis::tensor(client, shape.clone(), read, self.element);
                let full = self.axis.with_extent(n_fft);
                let padded = Axis::samples(&full, *dim, signal_len);
                let (_, host) = Axis::tensor(client, full, padded, self.element);
                (tensor, host)
            }
        }
    }

    fn bins(&self) -> usize {
        let len = match self.padded {
            Some(padding) => padding.n_fft,
            None => self.axis.shape[self.axis.dim],
        };
        len / 2 + 1
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
        let (signal, _) = case.signal(client);
        let spectrum = case.axis.with_extent(case.bins());
        let re = Axis::zeros(client, spectrum.clone(), case.element);
        let im = Axis::zeros(client, spectrum, case.element);
        let (signal, re_out, im_out) =
            (signal.binding(), re.clone().binding(), im.clone().binding());
        let launched = match case.padded {
            Some(padding) => rfft_launch_padded(
                client,
                signal,
                re_out,
                im_out,
                case.axis.dim,
                padding.signal_len,
                case.element,
            ),
            None => rfft_launch(client, signal, re_out, im_out, case.axis.dim, case.element),
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
        let (_, signal) = case.signal(&cubecl::test_device().client());
        let (re, im) = rfft_ref(&signal, case.axis.dim, None);
        vec![re, im]
    }

    /// Loose: a backend's `sin` and `cos` are less precise than the host's, as wgpu's are.
    fn tolerance(&self, _case: &Rfft) -> f32 {
        0.4
    }
}

/// The real inverse transform along an axis: of a seeded spectrum, or of `bins` bins the kernel
/// zero-pads to the bins of `n_fft` itself.
#[derive(Debug, Clone)]
pub(crate) struct Irfft {
    pub axis: Axis,
    pub padded_to: Option<usize>,
}

impl Irfft {
    pub(crate) fn seeded(shape: &[usize], dim: usize) -> Self {
        Self {
            axis: Axis::of(shape, dim),
            padded_to: None,
        }
    }

    /// The spectrum of `shape` along `dim`, zero-padded to the bins of `n_fft`.
    pub(crate) fn padded(shape: &[usize], dim: usize, n_fft: usize) -> Self {
        Self {
            axis: Axis::of(shape, dim),
            padded_to: Some(n_fft),
        }
    }

    /// The spectrum the kernel reads, real and imaginary, and the one the host transforms.
    fn spectrum(&self, client: &Client) -> [(TensorHandle, HostData); 2] {
        let element = f32::elem_type_native();
        let Axis { shape, dim } = &self.axis;
        match self.padded_to {
            None => [43, 44].map(|seed| Axis::seeded(client, shape.clone(), seed, element)),
            Some(n_fft) => [(); 2].map(|()| {
                let bins = shape[*dim];
                let read = Axis::samples(shape, *dim, bins);
                let (tensor, _) = Axis::tensor(client, shape.clone(), read, element);
                let full = self.axis.with_extent(n_fft / 2 + 1);
                let padded = Axis::samples(&full, *dim, bins);
                let (_, host) = Axis::tensor(client, full, padded, element);
                (tensor, host)
            }),
        }
    }

    fn signal_len(&self) -> usize {
        self.padded_to
            .unwrap_or((self.axis.shape[self.axis.dim] - 1) * 2)
    }
}

/// The real inverse transform's launch, against the host's.
pub(crate) struct IrfftLaunch;

impl ScopeUnderTest for IrfftLaunch {
    type Case = Irfft;

    fn launch(&self, client: &Client, case: &Irfft) -> Result<Vec<Written>, Refusal> {
        let element = f32::elem_type_native();
        let [(re, _), (im, _)] = case.spectrum(client);
        let signal = Axis::zeros(client, case.axis.with_extent(case.signal_len()), element);
        let (re, im, out) = (re.binding(), im.binding(), signal.clone().binding());
        let dim = case.axis.dim;
        let launched = match case.padded_to {
            Some(_) => irfft_launch_padded(client, re, im, out, dim, case.axis.shape[dim], element),
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
        let [(_, re), (_, im)] = case.spectrum(&cubecl::test_device().client());
        vec![irfft_ref(&re, &im, case.axis.dim, None)]
    }

    fn tolerance(&self, _case: &Irfft) -> f32 {
        0.01
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

    /// The input, real and imaginary, seeded.
    fn input(&self, client: &Client) -> [(TensorHandle, HostData); 2] {
        [42, 7].map(|seed| {
            Axis::seeded(
                client,
                self.axis.shape.clone(),
                seed,
                f32::elem_type_native(),
            )
        })
    }
}

/// The complex transform launched forward, then inverse, against the input it scales.
pub(crate) struct CfftThereAndBack;

impl ScopeUnderTest for CfftThereAndBack {
    type Case = CfftRoundTrip;

    fn launch(&self, client: &Client, case: &CfftRoundTrip) -> Result<Vec<Written>, Refusal> {
        let element = f32::elem_type_native();
        let [(input_re, _), (input_im, _)] = case.input(client);
        let shape = case.axis.shape.clone();
        let [spectrum_re, spectrum_im, recovered_re, recovered_im] =
            [(); 4].map(|()| Axis::zeros(client, shape.clone(), element));
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
        let n = case.axis.shape[case.axis.dim] as f32;
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
