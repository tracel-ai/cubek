//! What every convolution routine computes, on the host from the same inputs the routine reads:
//! the forward pass, the gradient of its input and the gradient of its weight, of any rank, grouped
//! or not, padded unevenly or not; and the scope a test reaches a routine through.
//!
//! Tensors lie as the routines take them, `[batch, ..spatial, channels]` in and out and
//! `[out_channels, ..kernel, channels / groups]` weights, in whichever storage the case states: the
//! gaps a storage leaves hold a poison no convolution here reaches, so a read past a tensor's
//! elements, or a cell the routine never writes, shows.

use core::fmt::Debug;

use cubecl::client::Client;
use cubecl::ir::{AddressType, ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::components::{
    ConvSetupError, ConvolutionOperation, ConvolutionProblem, Dimensionality,
};
use cubek_convolution::{ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::{InputBinding, MatrixLayout};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};
use half::{bf16, f16};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const BF16: ElemType = ElemType::Float(FloatKind::BF16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// What a gap of a stored tensor holds: no convolution here reaches it.
const POISON: f32 = 99.;

/// One pass of a convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    /// The output, from the input and the weight, plus the bias where the case has one.
    Forward,
    /// The input's gradient, from the output's gradient and the weight.
    BackwardData,
    /// The weight's gradient, from the input and the output's gradient.
    BackwardWeight,
}

impl Pass {
    pub(crate) const ALL: [Pass; 3] = [Pass::Forward, Pass::BackwardData, Pass::BackwardWeight];
}

/// How a tensor lies in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Storage {
    /// Contiguous, channels innermost.
    Dense,
    /// Channels outermost after the batch: what a framework's `NCHW` tensor is.
    ChannelsFirst,
    /// Channels innermost, each position's channels padded to a multiple of eight, with a guard
    /// past the last batch.
    Padded,
    /// Channels outermost after the batch, each row of the innermost spatial axis padded to a
    /// multiple of eight, with a guard past the last batch.
    ChannelsFirstPadded,
    /// Channels innermost, every element two apart.
    Gapped,
}

/// The values a case's tensors hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Values {
    /// Sixty-fourths in `[-0.52, 0.52]`: every element holds them exactly, and an `f32`
    /// accumulator sums their products exactly, so only the output's rounding differs.
    SixtyFourths,
    /// Small integers on a short period: every partial sum stays under 2048, so even an `f16`
    /// accumulator sums them exactly, and the routine must match the host exactly.
    Ramp,
}

/// One convolution of `N` spatial axes: its geometry, the element it reads and writes, how its
/// tensors lie and what they hold.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv<const N: usize> {
    batches: usize,
    spatial: [usize; N],
    channels: usize,
    out_channels: usize,
    groups: usize,
    kernel: [usize; N],
    stride: [usize; N],
    dilation: [usize; N],
    /// At the beginning and the end of each spatial axis. A routine is told only the beginning:
    /// the end is the larger output it is handed.
    padding: [(usize, usize); N],
    bias: bool,
    storage: Storage,
    out_grad_storage: Storage,
    element: ElemType,
    values: Values,
}

impl<const N: usize> Conv<N> {
    /// A convolution of `element`, one batch of one channel to one, a kernel of one, unpadded: the
    /// rest is stated by the builders.
    pub(crate) fn of(element: ElemType) -> Self {
        Self {
            batches: 1,
            spatial: [1; N],
            channels: 1,
            out_channels: 1,
            groups: 1,
            kernel: [1; N],
            stride: [1; N],
            dilation: [1; N],
            padding: [(0, 0); N],
            bias: false,
            storage: Storage::Dense,
            out_grad_storage: Storage::Dense,
            element,
            values: Values::SixtyFourths,
        }
    }

    pub(crate) fn batches(self, batches: usize) -> Self {
        Self { batches, ..self }
    }

    pub(crate) fn spatial(self, spatial: [usize; N]) -> Self {
        Self { spatial, ..self }
    }

    pub(crate) fn channels(self, channels: usize) -> Self {
        Self { channels, ..self }
    }

    pub(crate) fn out_channels(self, out_channels: usize) -> Self {
        Self {
            out_channels,
            ..self
        }
    }

    /// The same convolution in `groups` groups, each output channel reading its group's input
    /// channels alone.
    pub(crate) fn groups(self, groups: usize) -> Self {
        Self { groups, ..self }
    }

    pub(crate) fn kernel(self, kernel: [usize; N]) -> Self {
        Self { kernel, ..self }
    }

    pub(crate) fn stride(self, stride: [usize; N]) -> Self {
        Self { stride, ..self }
    }

    pub(crate) fn dilation(self, dilation: [usize; N]) -> Self {
        Self { dilation, ..self }
    }

    /// Padded `(beginning, end)` along each spatial axis.
    pub(crate) fn padding(self, padding: [(usize, usize); N]) -> Self {
        Self { padding, ..self }
    }

    /// Padded `padding` at both ends of each spatial axis.
    pub(crate) fn padded(self, padding: [usize; N]) -> Self {
        Self {
            padding: padding.map(|side| (side, side)),
            ..self
        }
    }

    /// The same convolution plus a bias, one value per output channel.
    pub(crate) fn with_bias(self) -> Self {
        Self { bias: true, ..self }
    }

    /// Every tensor of the convolution stored as `storage`.
    pub(crate) fn stored(self, storage: Storage) -> Self {
        Self {
            storage,
            out_grad_storage: storage,
            ..self
        }
    }

    /// The output's gradient stored as `storage`, whatever the other tensors are.
    pub(crate) fn out_grad_stored(self, storage: Storage) -> Self {
        Self {
            out_grad_storage: storage,
            ..self
        }
    }

    pub(crate) fn values(self, values: Values) -> Self {
        Self { values, ..self }
    }

    pub(crate) fn storage(&self) -> Storage {
        self.storage
    }

    pub(crate) fn element(&self) -> ElemType {
        self.element
    }

    pub(crate) fn group_count(&self) -> usize {
        self.groups
    }

    pub(crate) fn channels_count(&self) -> usize {
        self.channels
    }

    pub(crate) fn out_channels_count(&self) -> usize {
        self.out_channels
    }

    pub(crate) fn kernel_size(&self) -> [usize; N] {
        self.kernel
    }

    /// The extent of each spatial axis of the output.
    pub(crate) fn out_spatial(&self) -> [usize; N] {
        core::array::from_fn(|axis| {
            let (begin, end) = self.padding[axis];
            let reach = self.dilation[axis] * (self.kernel[axis] - 1) + 1;
            (self.spatial[axis] + begin + end - reach) / self.stride[axis] + 1
        })
    }

    /// What a routine is told: the stride, the beginning's padding and the dilation.
    pub(crate) fn args(&self) -> ConvolutionArgs<N> {
        ConvolutionArgs {
            stride: self.stride,
            padding: self.padding.map(|(begin, _)| begin),
            dilation: self.dilation,
        }
    }

    pub(crate) fn input_shape(&self) -> Vec<usize> {
        [
            vec![self.batches],
            self.spatial.to_vec(),
            vec![self.channels],
        ]
        .concat()
    }

    pub(crate) fn weight_shape(&self) -> Vec<usize> {
        let per_group = self.channels / self.groups;
        [
            vec![self.out_channels],
            self.kernel.to_vec(),
            vec![per_group],
        ]
        .concat()
    }

    pub(crate) fn out_shape(&self) -> Vec<usize> {
        let out = self.out_spatial().to_vec();
        [vec![self.batches], out, vec![self.out_channels]].concat()
    }

    /// The shape of what `pass` writes.
    pub(crate) fn written_shape(&self, pass: Pass) -> Vec<usize> {
        match pass {
            Pass::Forward => self.out_shape(),
            Pass::BackwardData => self.input_shape(),
            Pass::BackwardWeight => self.weight_shape(),
        }
    }

    /// `count` values of the kind the case holds, for the tensor `role` names, as the case's
    /// element holds them.
    fn host_values(&self, count: usize, role: Role) -> Vec<f32> {
        let (seed, period) = match role {
            Role::Input => (3, 4),
            Role::Weight => (23, 3),
            Role::OutGrad => (41, 6),
            Role::Bias => (53, 5),
        };
        (0..count)
            .map(|i| match self.values {
                Values::SixtyFourths => ((i * 17 + seed) % 67) as f32 / 64. - 33. / 64.,
                Values::Ramp => (i % period) as f32 - 1.,
            })
            .map(|value| match self.element {
                F16 => f16::from_f32(value).to_f32(),
                BF16 => bf16::from_f32(value).to_f32(),
                _ => value,
            })
            .collect()
    }

    pub(crate) fn input(&self) -> Vec<f32> {
        self.host_values(self.input_shape().iter().product(), Role::Input)
    }

    pub(crate) fn weight(&self) -> Vec<f32> {
        self.host_values(self.weight_shape().iter().product(), Role::Weight)
    }

    pub(crate) fn out_grad(&self) -> Vec<f32> {
        self.host_values(self.out_shape().iter().product(), Role::OutGrad)
    }

    pub(crate) fn bias_values(&self) -> Vec<f32> {
        self.host_values(self.out_channels, Role::Bias)
    }

    /// The three passes on the host, in `f64`, by walking every window of the forward pass once:
    /// the output, the input's gradient and the weight's gradient, each row-major.
    fn passes(&self) -> [Vec<f64>; 3] {
        let (input, weight, out_grad) = (self.input(), self.weight(), self.out_grad());
        let out_spatial = self.out_spatial();
        let taps: usize = self.kernel.iter().product();
        let per_group = self.channels / self.groups;
        let out_per_group = self.out_channels / self.groups;
        let mut out = vec![0.0f64; out_grad.len()];
        let mut in_grad = vec![0.0f64; input.len()];
        let mut weight_grad = vec![0.0f64; weight.len()];
        for (at, y) in out.iter_mut().enumerate() {
            let co = at % self.out_channels;
            let group = co / out_per_group;
            let mut batch = at / self.out_channels;
            let mut position = [0; N];
            for axis in (0..N).rev() {
                position[axis] = batch % out_spatial[axis];
                batch /= out_spatial[axis];
            }
            for tap in 0..taps {
                let mut rest = tap;
                let mut read = [0; N];
                let mut inside = true;
                for axis in (0..N).rev() {
                    let k = rest % self.kernel[axis];
                    rest /= self.kernel[axis];
                    let p = (position[axis] * self.stride[axis] + k * self.dilation[axis]) as isize
                        - self.padding[axis].0 as isize;
                    inside &= p >= 0 && p < self.spatial[axis] as isize;
                    read[axis] = p.max(0) as usize;
                }
                if !inside {
                    continue;
                }
                let mut pixel = batch;
                for (&extent, &at) in self.spatial.iter().zip(&read) {
                    pixel = pixel * extent + at;
                }
                for ci in 0..per_group {
                    let xi = pixel * self.channels + group * per_group + ci;
                    let wi = (co * taps + tap) * per_group + ci;
                    *y += input[xi] as f64 * weight[wi] as f64;
                    in_grad[xi] += out_grad[at] as f64 * weight[wi] as f64;
                    weight_grad[wi] += out_grad[at] as f64 * input[xi] as f64;
                }
            }
        }
        if self.bias {
            let bias = self.bias_values();
            for (at, y) in out.iter_mut().enumerate() {
                *y += bias[at % self.out_channels] as f64;
            }
        }
        [out, in_grad, weight_grad]
    }

    /// What `pass` writes, on the host.
    pub(crate) fn reference(&self, pass: Pass) -> HostData {
        let [out, in_grad, weight_grad] = self.passes();
        let written = match pass {
            Pass::Forward => out,
            Pass::BackwardData => in_grad,
            Pass::BackwardWeight => weight_grad,
        };
        let values = written.into_iter().map(|value| value as f32).collect();
        HostData::row_major(
            HostDataVec::F32(values),
            Shape::from(self.written_shape(pass)),
        )
    }

    /// Exact where every accumulator sums the values exactly; else the output element's rounding.
    pub(crate) fn tolerance(&self) -> f32 {
        match (self.values, self.element) {
            (Values::Ramp, _) => 0.,
            (_, F16) => 2e-3,
            (_, BF16) => 1.6e-2,
            _ => 2e-4,
        }
    }

    /// `values`, logically row-major in `shape`, uploaded as `storage` lays them, its gaps poisoned.
    pub(crate) fn store(
        &self,
        client: &Client,
        shape: &[usize],
        values: &[f32],
        storage: Storage,
    ) -> TensorHandle {
        let rank = shape.len();
        let order: Vec<usize> = match storage {
            Storage::ChannelsFirst | Storage::ChannelsFirstPadded if rank > 2 => {
                [vec![0, rank - 1], (1..rank - 1).collect()].concat()
            }
            _ => (0..rank).collect(),
        };
        let mut strides = vec![0; rank];
        let mut size = match storage {
            Storage::Gapped => 2,
            _ => 1,
        };
        let padded = matches!(storage, Storage::Padded | Storage::ChannelsFirstPadded) && rank > 1;
        for &axis in order.iter().rev() {
            strides[axis] = size;
            size *= shape[axis];
            if padded && Some(&axis) == order.last() {
                size = size.next_multiple_of(8);
            }
        }
        if padded {
            size += strides[0];
        }
        let mut data = vec![POISON; size];
        for (index, value) in values.iter().enumerate() {
            let mut rest = index;
            let mut offset = 0;
            for axis in (0..rank).rev() {
                offset += rest % shape[axis] * strides[axis];
                rest /= shape[axis];
            }
            data[offset] = *value;
        }
        let bytes = match self.element {
            F16 => {
                f16::as_bytes(&data.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).to_vec()
            }
            BF16 => bf16::as_bytes(&data.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>())
                .to_vec(),
            _ => f32::as_bytes(&data).to_vec(),
        };
        TensorHandle::new(
            client.create_from_slice(&bytes),
            Shape::from(shape.to_vec()),
            Strides::new(&strides),
            self.element,
        )
    }

    /// Every tensor a pass of this convolution reads or writes, uploaded.
    pub(crate) fn tensors(&self, client: &Client, pass: Pass) -> ConvTensors {
        let written_shape = self.written_shape(pass);
        let poison = vec![POISON; written_shape.iter().product()];
        ConvTensors {
            input: self.store(client, &self.input_shape(), &self.input(), self.storage),
            weight: self.store(client, &self.weight_shape(), &self.weight(), self.storage),
            // A bias beside channels-first tensors is a strided view, its gaps poisoned.
            bias: self.bias.then(|| {
                let storage = match self.storage {
                    Storage::Dense | Storage::Padded => Storage::Dense,
                    _ => Storage::Gapped,
                };
                self.store(client, &[self.out_channels], &self.bias_values(), storage)
            }),
            out_grad: self.store(
                client,
                &self.out_shape(),
                &self.out_grad(),
                self.out_grad_storage,
            ),
            written: self.store(client, &written_shape, &poison, self.storage),
        }
    }

    /// The problem the multi-level routines plan `pass` as, its operands the tensors given.
    pub(crate) fn problem(&self, pass: Pass, tensors: &ConvTensors) -> ConvolutionProblem {
        let batches = self.batches;
        let taps: usize = self.kernel.iter().product();
        let in_positions: usize = self.spatial.iter().product();
        let out_positions: usize = self.out_spatial().iter().product();
        let (m, n, k, lhs, rhs, operation, padded_channels) = match pass {
            Pass::Forward => (
                batches * out_positions,
                self.out_channels,
                taps * self.channels,
                (MatrixLayout::RowMajor, None),
                (MatrixLayout::ColMajor, None),
                ConvolutionOperation::Forward,
                self.channels,
            ),
            Pass::BackwardData => (
                batches * in_positions,
                self.channels,
                self.out_channels * taps,
                (MatrixLayout::RowMajor, Some(&tensors.out_grad)),
                (MatrixLayout::RowMajor, Some(&tensors.weight)),
                ConvolutionOperation::BackwardData,
                self.out_channels,
            ),
            Pass::BackwardWeight => (
                self.out_channels,
                self.channels * taps,
                batches * out_positions,
                (MatrixLayout::ColMajor, Some(&tensors.input)),
                (MatrixLayout::RowMajor, Some(&tensors.out_grad)),
                ConvolutionOperation::BackwardWeight,
                self.channels,
            ),
        };
        let strides = |(layout, tensor): (MatrixLayout, Option<&TensorHandle>),
                       dims: [usize; 2]| {
            tensor.map_or_else(
                || layout.to_strides(&dims),
                |tensor| tensor.strides().clone(),
            )
        };
        ConvolutionProblem {
            m,
            n,
            k,
            lhs_strides: strides(lhs, [m, k]),
            rhs_strides: strides(rhs, [k, n]),
            lhs_layout: lhs.0,
            rhs_layout: rhs.0,
            kernel_size: self.kernel.iter().map(|&k| k as u32).collect(),
            stride: self.stride.iter().map(|&s| s as u32).collect(),
            padding: self
                .padding
                .iter()
                .map(|&(begin, _)| begin as i32)
                .collect(),
            dilation: self.dilation.iter().map(|&d| d as u32).collect(),
            batches,
            in_shape: Shape::from(self.spatial.to_vec()),
            channels: self.channels,
            out_channels: self.out_channels,
            padded_channels,
            out_shape: Shape::from(self.out_spatial().to_vec()),
            dimensionality: Self::dimensionality(),
            operation,
            global_dtypes: self.global_elems(),
            address_type: AddressType::U32,
        }
    }

    pub(crate) fn dimensionality() -> Dimensionality {
        match N {
            1 => Dimensionality::Dim1,
            2 => Dimensionality::Dim2,
            3 => Dimensionality::Dim3,
            _ => panic!("a convolution has one to three spatial axes, not {N}"),
        }
    }

    pub(crate) fn global_elems(&self) -> MatmulGlobalElems {
        MatmulGlobalElems {
            lhs: self.element,
            rhs: self.element,
            out: self.element,
        }
    }

    /// The elements a routine planned on the matmul reads and sums at.
    pub(crate) fn matmul_elems(&self) -> MatmulElems {
        MatmulElems::from_globals(&self.global_elems())
    }
}

/// Which tensor a case's values are for.
#[derive(Debug, Clone, Copy)]
enum Role {
    Input,
    Weight,
    OutGrad,
    Bias,
}

/// Every tensor a pass reads, and the one it writes, poisoned.
pub(crate) struct ConvTensors {
    pub input: TensorHandle,
    pub weight: TensorHandle,
    pub bias: Option<TensorHandle>,
    pub out_grad: TensorHandle,
    pub written: TensorHandle,
}

/// A convolution routine, launched on one pass of a case.
pub(crate) trait ConvRoutine<const N: usize>: Debug {
    /// Launch `pass` of `conv` on `tensors`, writing `tensors.written`.
    ///
    /// # Errors
    ///
    /// Why the routine runs no such pass.
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError>;
}

/// Which refusals a scope may answer a case with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declines {
    /// None: the routine serves every case it is given.
    Never,
    /// Its own, where the device runs no such pass.
    AsItStates,
}

/// One pass of one convolution.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvCase<const N: usize> {
    pub conv: Conv<N>,
    pub pass: Pass,
}

/// A convolution routine against the host's passes, on convolutions of `N` spatial axes.
pub(crate) struct Convolves<R, const N: usize> {
    routine: R,
    declines: Declines,
}

impl<R, const N: usize> Convolves<R, N> {
    pub(crate) fn new(routine: R, declines: Declines) -> Self {
        Self { routine, declines }
    }

    /// Each of `passes` of each of `convs`.
    pub(crate) fn at(&self, passes: &[Pass], convs: &[Conv<N>]) -> Vec<ConvCase<N>> {
        convs
            .iter()
            .flat_map(|&conv| passes.iter().map(move |&pass| ConvCase { conv, pass }))
            .collect()
    }
}

impl<R: ConvRoutine<N>, const N: usize> ScopeUnderTest for Convolves<R, N> {
    type Case = ConvCase<N>;

    /// # Panics
    ///
    /// Where a routine that declines nothing refuses a case.
    fn launch(&self, client: &Client, case: &ConvCase<N>) -> Result<Vec<Written>, Refusal> {
        let tensors = case.conv.tensors(client, case.pass);
        match (
            self.routine.launch(client, &case.conv, case.pass, &tensors),
            self.declines,
        ) {
            (Ok(()), _) => Ok(vec![Written {
                name: "written",
                tensor: tensors.written,
                host: HostDataType::F32,
            }]),
            (Err(_), Declines::AsItStates) => Err(Refusal::Declined {
                why: "the routine runs no such pass on this device",
            }),
            (Err(why), Declines::Never) => {
                panic!("{:?} refused {case:?}: {why:?}", self.routine)
            }
        }
    }

    fn reference(&self, case: &ConvCase<N>) -> Vec<HostData> {
        vec![case.conv.reference(case.pass)]
    }

    fn tolerance(&self, case: &ConvCase<N>) -> f32 {
        case.conv.tolerance()
    }
}

/// A public [`Strategy`] reached through [`launch_ref`], as burn reaches it, on every pass.
#[derive(Debug)]
pub(crate) struct ByStrategy(pub Strategy);

impl<const N: usize> ConvRoutine<N> for ByStrategy {
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        launch_ref(
            &self.0,
            client,
            tensors.inputs(conv, pass),
            conv.args(),
            conv.matmul_elems(),
        )
    }
}

impl ConvTensors {
    /// The bindings `pass` reads and writes, as [`launch_ref`] takes them.
    pub(crate) fn inputs<const N: usize>(&self, conv: &Conv<N>, pass: Pass) -> ConvolutionInputs {
        let read =
            |tensor: &TensorHandle| InputBinding::new(tensor.clone().binding(), conv.element);
        let written = self.written.clone().binding();
        match pass {
            Pass::Forward => ConvolutionInputs::Forward {
                input: read(&self.input),
                weight: read(&self.weight),
                bias: self.bias.as_ref().map(read),
                out: written,
            },
            Pass::BackwardData => ConvolutionInputs::BackwardData {
                out_grad: read(&self.out_grad),
                weights: read(&self.weight),
                in_grad: written,
            },
            Pass::BackwardWeight => ConvolutionInputs::BackwardWeight {
                input: read(&self.input),
                out_grad: read(&self.out_grad),
                weight_grad: written,
            },
        }
    }
}
