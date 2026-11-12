//! How a packed tensor binds: what it counts, and the stacks of slabs it refuses.

use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_matmul::kernels::select::packed::nvfp4_scheme;
use cubek_matmul::kernels::tensor::{PACKED_STACK_UNREAD, PackedTensor, PlainTensor, StoredTensor};
use cubek_microkernel::matmul::MatmulOperand;
use cubek_std::layout::stored::Operand;
use cubek_std::matmul_axes::{K, N};

use crate::tiled::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec, block_of};

/// A packed tensor counts the bytes its words and scales bind, not the buffer they share: a
/// framework stores the three in one, each bound over its own region of it. An autotune's
/// roofline reads these bytes, and counting the shared buffer once a part tripled them.
#[test]
fn a_packed_tensor_binds_its_parts_not_the_buffer_they_share() {
    let client = cubek_test_utils::client();
    let scheme = nvfp4_scheme();
    let (rows, k) = (16, 64);
    let word = ElemType::UInt(UIntKind::U32).size();
    let words = rows * k / scheme.num_quants();
    let scale_words = rows * k / block_of(&scheme) / word;
    let total = (words + scale_words + 1) * word;
    let buffer = client.empty(total);
    let part = |from: usize, words: usize, shape: [usize; 2], elem: ElemType| {
        let handle = buffer
            .clone()
            .offset_start(from as u64)
            .offset_end((total - from - words * word) as u64);
        PlainTensor::of_handle(TensorHandle::new_contiguous(
            Shape::from(shape.to_vec()),
            handle,
            elem,
        ))
    };
    let u32 = ElemType::UInt(UIntKind::U32);
    let packed = PackedTensor::new(
        part(0, words, [rows, words / rows], u32),
        part(words * word, scale_words, [rows, scale_words / rows], u32),
        Some(part((words + scale_words) * word, 1, [1, 1], u32)),
        scheme,
        Operand::of_scheme(&scheme, K, [N, K]).expect("one block along k"),
        [(N, rows), (K, k)],
    );

    assert_eq!(StoredTensor::Packed(packed).bytes(), total);
}

/// Slabs along `K` that split a scale block, or a word of block scales, are refused rather than
/// read across.
#[test]
fn a_packed_stack_whose_slabs_split_a_scale_block_is_refused() {
    let client = cubecl::test_device().client();
    let along_k = |format, slabs| {
        let matrix = PackedSlabs::new(
            &client,
            PackedSlabsSpec {
                format,
                operand: MatmulOperand::Rhs,
                slabs: 1,
                rows: 8,
                k: 128,
                seed: 5,
            },
        );
        matrix.matrix().stacked(slabs, K).err()
    };
    // 32-value blocks: 4 slabs hold whole blocks, 8 hold half of one.
    assert_eq!(along_k(PackedFormat::Q8, 4), None);
    assert_eq!(along_k(PackedFormat::Q8, 8), Some(PACKED_STACK_UNREAD));
    // NVFP4: four 16-value blocks a word of scales, so 4 slabs split a word.
    assert_eq!(along_k(PackedFormat::Nvfp4, 2), None);
    assert_eq!(along_k(PackedFormat::Nvfp4, 4), Some(PACKED_STACK_UNREAD));
}
