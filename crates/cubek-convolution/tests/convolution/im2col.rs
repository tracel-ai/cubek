//! Explicit lowering, checked independently of any convolution or matmul: every window of the
//! input written as a row of columns, a tap outside the input written as zero.

use cubecl::client::Client;
use cubecl::ir::FloatKind;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_convolution::{ConvolutionArgs, Im2colTensors, launch_im2col};
use cubek_test_utils::{HostData, HostDataType, HostDataVec, ScopeUnderTest, Written};

use crate::convolution::{Conv, F32, Storage};

/// The columns of one convolution's input, its rows `gap` elements apart within a padded row.
#[derive(Debug, Clone, Copy)]
struct Lowering<const N: usize> {
    conv: Conv<N>,
    gap: usize,
}

impl<const N: usize> Lowering<N> {
    /// Each of `convs` read channels last and channels first into dense columns, and read gapped
    /// into gapped columns.
    fn every_storage_of(convs: &[Conv<N>]) -> Vec<Self> {
        convs
            .iter()
            .flat_map(|&conv| {
                [
                    Self {
                        conv: conv.stored(Storage::Padded),
                        gap: 1,
                    },
                    Self {
                        conv: conv.stored(Storage::ChannelsFirst),
                        gap: 1,
                    },
                    Self {
                        conv: conv.stored(Storage::Gapped),
                        gap: 2,
                    },
                ]
            })
            .collect()
    }

    /// One row per output position, one column per tap and channel.
    fn extents(&self) -> (usize, usize) {
        let rows = self.conv.out_shape()[..N + 1].iter().product();
        let width = self.conv.channels_count() * self.conv.kernel_size().iter().product::<usize>();
        (rows, width)
    }
}

/// The im2col kernel against the host's columns, on convolutions of `N` spatial axes.
struct Im2col<const N: usize>;

impl<const N: usize> ScopeUnderTest for Im2col<N> {
    type Case = Lowering<N>;

    fn launch(&self, client: &Client, lowering: &Lowering<N>) -> Result<Vec<Written>, Refusal> {
        let conv = &lowering.conv;
        let (rows, width) = lowering.extents();
        let row_stride = (width * lowering.gap + 1).next_multiple_of(16);
        let input = conv.store(client, &conv.input_shape(), &conv.input(), conv.storage());
        let columns = TensorHandle::new(
            client.create_from_slice(f32::as_bytes(&vec![99.0; rows * row_stride])),
            [rows, width],
            [row_stride, lowering.gap],
            f32::elem_type_native(),
        );
        launch_im2col(
            client,
            Im2colTensors {
                input: input.binding(),
                columns: columns.clone().binding(),
            },
            conv.out_spatial(),
            conv.kernel_size(),
            conv.args(),
            f32::elem_type_native(),
        )
        .unwrap_or_else(|why| panic!("{lowering:?} refused: {why:?}"));
        Ok(vec![Written {
            name: "columns",
            tensor: columns,
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, lowering: &Lowering<N>) -> Vec<HostData> {
        let conv = &lowering.conv;
        let (rows, width) = lowering.extents();
        let input = conv.input();
        let spatial = &conv.input_shape()[1..N + 1];
        let out_spatial = conv.out_spatial();
        let channels = conv.channels_count();
        let kernel = conv.kernel_size();
        let args = conv.args();
        let mut columns = vec![0.0; rows * width];
        for row in 0..rows {
            let mut batch = row;
            let mut position = [0; N];
            for axis in (0..N).rev() {
                position[axis] = batch % out_spatial[axis];
                batch /= out_spatial[axis];
            }
            for column in 0..width {
                let channel = column % channels;
                let mut tap = column / channels;
                let mut read = [0; N];
                let mut inside = true;
                for axis in (0..N).rev() {
                    let k = tap % kernel[axis];
                    tap /= kernel[axis];
                    let p = (position[axis] * args.stride[axis] + k * args.dilation[axis]) as isize
                        - args.padding[axis] as isize;
                    inside &= p >= 0 && p < spatial[axis] as isize;
                    read[axis] = p.max(0) as usize;
                }
                if inside {
                    let mut pixel = batch;
                    for (&extent, &at) in spatial.iter().zip(&read) {
                        pixel = pixel * extent + at;
                    }
                    columns[row * width + column] = input[pixel * channels + channel];
                }
            }
        }
        vec![HostData::row_major(
            HostDataVec::F32(columns),
            Shape::from(vec![rows, width]),
        )]
    }

    /// A copy: exact.
    fn tolerance(&self, _lowering: &Lowering<N>) -> f32 {
        0.
    }
}

#[test]
fn im2col_matches_reference() {
    Im2col.check(Lowering::every_storage_of(&[Conv::of(F32)
        .batches(2)
        .spatial([13])
        .channels(3)
        .kernel([3])
        .stride([2])
        .dilation([2])
        .padding([(1, 2)])]));
    Im2col.check(Lowering::every_storage_of(&[
        Conv::of(F32)
            .batches(1)
            .spatial([1, 1])
            .channels(4)
            .kernel([3, 3])
            .stride([1, 1])
            .dilation([1, 1])
            .padding([(2, 1), (1, 2)]),
        Conv::of(F32)
            .batches(2)
            .spatial([13, 17])
            .channels(4)
            .kernel([3, 3])
            .stride([1, 1])
            .dilation([1, 1])
            .padding([(1, 1), (1, 1)]),
        Conv::of(F32)
            .batches(3)
            .spatial([9, 11])
            .channels(17)
            .kernel([2, 3])
            .stride([2, 1])
            .dilation([2, 2])
            .padding([(0, 1), (3, 0)]),
        Conv::of(F32)
            .batches(2)
            .spatial([5, 7])
            .channels(32)
            .kernel([3, 2])
            .stride([2, 2])
            .dilation([1, 2])
            .padding([(1, 0), (0, 2)]),
    ]));
    Im2col.check(Lowering::every_storage_of(&[Conv::of(F32)
        .batches(1)
        .spatial([4, 5, 7])
        .channels(4)
        .kernel([2, 3, 2])
        .stride([1, 2, 1])
        .dilation([2, 1, 2])
        .padding([(1, 0), (0, 1), (2, 0)])]));
}

/// An element other than `f32`, a zero stride and columns of the wrong width are declined.
#[test]
fn im2col_declines_unsupported_inputs() {
    let client = &cubecl::test_device().client();
    let dtype = ElemType::Float(FloatKind::F16);
    let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
    let columns = TensorHandle::empty(client, [25, 36], dtype);
    assert!(
        launch_im2col(
            client,
            Im2colTensors {
                input: input.binding(),
                columns: columns.binding()
            },
            [5, 5],
            [3, 3],
            ConvolutionArgs {
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1]
            },
            dtype,
        )
        .is_err()
    );
    for stride in [[0, 1], [1, 1]] {
        let dtype = f32::elem_type_native();
        let input = TensorHandle::empty(client, [1, 5, 5, 4], dtype);
        let columns = TensorHandle::empty(client, [25, 35], dtype);
        assert!(
            launch_im2col(
                client,
                Im2colTensors {
                    input: input.binding(),
                    columns: columns.binding()
                },
                [5, 5],
                [3, 3],
                ConvolutionArgs {
                    stride,
                    padding: [1, 1],
                    dilation: [1, 1]
                },
                dtype,
            )
            .is_err()
        );
    }
}
