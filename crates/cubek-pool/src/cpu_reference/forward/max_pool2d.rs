use cubecl::zspace::{Shape, Strides};
use cubek_test_utils::{HostData, HostDataVec};

pub fn cpu_reference_max_pool2d(
    input: &HostData,
    input_shape: [usize; 4],
    output_shape: [usize; 4],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> HostData {
    let [batch_size, in_h, in_w, channels] = input_shape;

    let out_h = output_shape[1];
    let out_w = output_shape[2];

    let output_shape = [batch_size, out_h, out_w, channels];
    let mut output = Vec::with_capacity(output_shape.iter().product());

    for batch in 0..batch_size {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for channel in 0..channels {
                    let mut max_value = f32::NEG_INFINITY;

                    for kernel_h in 0..kernel_size[0] {
                        let ih = oh as isize * stride[0] as isize
                            + kernel_h as isize * dilation[0] as isize
                            - padding[0] as isize;

                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }

                        for kernel_w in 0..kernel_size[1] {
                            // Apply Dilation to the width offset
                            let iw = ow as isize * stride[1] as isize
                                + kernel_w as isize * dilation[1] as isize
                                - padding[1] as isize;

                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }

                            let value = input.get_f32(&[batch, ih as usize, iw as usize, channel]);
                            if value > max_value {
                                max_value = value;
                            }
                        }
                    }

                    output.push(max_value);
                }
            }
        }
    }

    HostData {
        data: HostDataVec::F32(output),
        shape: Shape::from(output_shape.to_vec()), // Must return output shape
        strides: Strides::new(&row_major_strides(&output_shape)),
    }
}

fn row_major_strides(shape: &[usize; 4]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for index in (0..shape.len() - 1).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }

    strides
}
