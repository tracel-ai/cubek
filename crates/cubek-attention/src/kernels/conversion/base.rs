//! An operand converted to another element ahead of a launch, into a buffer of its own: what a
//! routine reads at an element its stage cannot decode it into, where the operand is small
//! enough that one pass over it is nothing beside the attention. The query of a prompt chunk read
//! at its keys' `e4m3` is one: a few megabytes, against a cache of gigabytes.

use super::kernel::converted_kernel;
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;

/// `tensor` converted to `element`, every value cast, into a buffer of its own that holds once
/// each value of `tensor` it reaches: contiguous, except that a dim `tensor` repeats at stride
/// zero is repeated the same way over the converted buffer. Queries packed end to end and viewed
/// once per sequence, the batch at stride zero, convert as the packed queries, not once per
/// sequence.
pub(crate) fn converted(client: &Client, tensor: &PlainTensor, element: ElemType) -> PlainTensor {
    let (shape, strides) = (&tensor.binding.shape, &tensor.binding.strides);
    let held = held_once(shape, strides);
    let elements: usize = held.iter().product();
    let (_, converted) = PlainTensor::empty_contiguous(client, held.clone().into(), element);
    if elements > 0 {
        let source = tensor.viewed(Metadata::new(held, strides.clone()));
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = calculate_cube_count_elemwise(client, elements, cube_dim);
        // SAFETY: every unit past `elements` returns before it reads; each reads `source` at its
        // own coordinate through its strides and writes its own element of `converted`.
        unsafe {
            converted_kernel::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                source.binding.clone().into_tensor_arg(),
                converted.binding.clone().into_tensor_arg(),
                u32::try_from(elements).expect("a converted operand counts its elements in a u32"),
                [tensor.elem, element],
            );
        }
    }
    let repeated = strides
        .iter()
        .zip(converted.binding.strides.iter())
        .map(|(&stride, &held)| match stride {
            0 => 0,
            _ => held,
        })
        .collect::<Vec<_>>();
    converted.viewed(Metadata::new(shape.clone(), repeated))
}

/// The shape of what a view at `shape` and `strides` holds once: each dim at stride zero, which
/// repeats the same values, one long.
fn held_once(shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(strides)
        .map(|(&extent, &stride)| match stride {
            0 => extent.min(1),
            _ => extent,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{converted, held_once};
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::zspace::Shape;
    use cubecl::zspace::metadata::Metadata;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_test_utils::{HostData, HostDataType, TestInput};

    /// Packed queries `[Σq, h, d]` viewed once per sequence, `[b, h, Σq, d]` with the batch at
    /// stride zero, convert into a buffer of the packed queries' size, and read back through the
    /// same stride-zero batch, every sequence's view holds every value converted.
    #[test]
    fn a_stride_zero_view_converts_its_buffer_once() {
        let (sequences, queries, heads, head) = (3, 5, 4, 8);
        let client: Client = cubecl::test_device().client();
        let (f16, f32) = (half::f16::elem_type_native(), f32::elem_type_native());
        let data: Vec<f32> = (0..queries * heads * head)
            .map(|i| (i % 61) as f32 - 30.)
            .collect();
        let packed = PlainTensor::of_handle(
            TestInput::builder(client.clone(), Shape::from(vec![queries, heads, head]))
                .dtype(f16)
                .custom(data.clone())
                .generate_without_host_data(),
        );
        let view = packed.viewed(Metadata::new(
            [sequences, heads, queries, head],
            [0, head, heads * head, 1],
        ));
        let out = converted(&client, &view, f32);
        // The packed queries' bytes, up to the allocator's alignment: never one copy a sequence.
        let packed_bytes = queries * heads * head * f32.size();
        assert!(
            (packed_bytes..2 * packed_bytes).contains(&out.bytes()),
            "{} bytes converted, the packed queries being {packed_bytes}",
            out.bytes()
        );
        assert_eq!(
            out.binding.shape.as_slice(),
            [sequences, heads, queries, head]
        );
        assert_eq!(out.binding.strides[0], 0);
        let read = cubecl::std::tensor::into_contiguous(&client, out.binding.clone(), out.elem);
        let read = HostData::from_tensor_handle(&client, read, HostDataType::F32);
        for sequence in 0..sequences {
            for h in 0..heads {
                for q in 0..queries {
                    for c in 0..head {
                        assert_eq!(
                            read.get_f32(&[sequence, h, q, c]),
                            data[(q * heads + h) * head + c],
                            "sequence {sequence} head {h} query {q} [{c}]"
                        );
                    }
                }
            }
        }
    }

    /// Devstral's query, 32 heads of 128 over 8 key heads, for 64 sequences of 512 queries packed
    /// end to end, `[32768, 32, 128]`, as the attention views it, `[b, kv, g, Σq, d]` with the
    /// batch at stride zero: a logical `2^33` values, past a `u32`, of which the conversion holds
    /// the `2^27` the packed queries are.
    #[test]
    fn packed_queries_convert_once() {
        let (sequences, queries, heads, kv_heads, head) = (64, 64 * 512, 32, 8, 128);
        let group = heads / kv_heads;
        let shape = [sequences, kv_heads, group, queries, head];
        let strides = [0, group * head, head, heads * head, 1];
        assert_eq!(shape.iter().product::<usize>(), 1 << 33);
        let held = held_once(&shape, &strides);
        assert_eq!(held, [1, kv_heads, group, queries, head]);
        let elements: usize = held.iter().product();
        assert_eq!(elements, queries * heads * head);
        assert!(u32::try_from(elements).is_ok());
    }

    /// A view that repeats nothing is held whole, an empty one holding nothing.
    #[test]
    fn a_view_without_repeats_is_held_whole() {
        assert_eq!(held_once(&[2, 3, 4], &[12, 4, 1]), [2, 3, 4]);
        assert_eq!(held_once(&[0, 3], &[0, 1]), [0, 3]);
    }
}
