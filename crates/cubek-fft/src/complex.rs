//! Modified from tensor4all/cubek `02efeba9` (MIT OR Apache-2.0):
//! nongeneric CubeCL handles and explicit scalar-stride/offset validation.

use cubecl::{
    prelude::{Client, ElemType, Scalar, TensorBinding},
    server::Handle,
    std::tensor::TensorHandle,
};

use crate::FftError;

/// A C32 tensor represented as adjacent real and imaginary F32 scalars.
///
/// Shapes and logical strides are measured in complex elements; the underlying
/// CubeCL tensor carries scalar strides (`2 * logical_stride`), so the
/// imaginary component of a logical element lives at `scalar_index + 1`.
#[derive(Clone)]
pub struct ComplexTensorHandle {
    tensor: TensorHandle,
    logical_strides: Vec<usize>,
    physical_scalar_len: usize,
}

impl ComplexTensorHandle {
    /// Allocates a contiguous C32 tensor with the requested logical shape.
    pub fn empty(client: &Client, shape: Vec<usize>) -> Result<Self, FftError> {
        let dtype = f32::elem_type_native();
        let logical_strides = contiguous_strides(&shape)?;
        let (_, physical_scalar_len) = scalar_layout(&shape, &logical_strides)?;
        let byte_len = physical_scalar_len
            .checked_mul(dtype.size())
            .ok_or(FftError::SizeOverflow)?;

        Self::new_strided(shape, logical_strides, client.empty(byte_len), dtype)
    }

    /// Wraps a contiguous C32 buffer whose logical elements occupy adjacent scalar pairs.
    pub fn new_contiguous(
        shape: Vec<usize>,
        handle: Handle,
        dtype: ElemType,
    ) -> Result<Self, FftError> {
        let logical_strides = contiguous_strides(&shape)?;
        Self::new_strided(shape, logical_strides, handle, dtype)
    }

    /// Wraps a C32 buffer using logical-complex-element strides.
    pub fn new_strided(
        shape: Vec<usize>,
        logical_strides: Vec<usize>,
        handle: Handle,
        dtype: ElemType,
    ) -> Result<Self, FftError> {
        ensure_c32_dtype(dtype)?;
        if shape.len() != logical_strides.len() {
            return Err(FftError::RankMismatch {
                shape_rank: shape.len(),
                stride_rank: logical_strides.len(),
            });
        }

        let offset = handle.offset_start.unwrap_or_default();
        let offset_end = handle.offset_end.unwrap_or_default();
        let used_bytes = handle
            .size()
            .checked_sub(offset)
            .and_then(|remaining| remaining.checked_sub(offset_end))
            .ok_or(FftError::InvalidBufferRange {
                size: handle.size(),
                offset_start: offset,
                offset_end,
            })?;
        if !offset.is_multiple_of(dtype.size() as u64) {
            return Err(FftError::MisalignedBuffer {
                offset,
                scalar_size: dtype.size(),
            });
        }

        let (scalar_strides, physical_scalar_len) = scalar_layout(&shape, &logical_strides)?;
        let available = usize::try_from(used_bytes / dtype.size() as u64)
            .map_err(|_| FftError::SizeOverflow)?;
        if available < physical_scalar_len {
            return Err(FftError::InsufficientBuffer {
                required: physical_scalar_len,
                available,
            });
        }

        Ok(Self {
            tensor: TensorHandle::new(handle, shape, scalar_strides, dtype),
            logical_strides,
            physical_scalar_len,
        })
    }

    /// Returns the logical complex shape.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Returns strides measured in logical complex elements.
    pub fn strides(&self) -> &[usize] {
        &self.logical_strides
    }

    /// Returns physical strides measured in F32 scalars.
    pub fn scalar_strides(&self) -> &[usize] {
        self.tensor.strides()
    }

    /// Returns the number of scalar F32 elements reachable through this layout.
    pub fn physical_scalar_len(&self) -> usize {
        self.physical_scalar_len
    }

    /// Returns the physical scalar storage type.
    pub fn dtype(&self) -> ElemType {
        self.tensor.dtype
    }

    /// Borrows the handle for a later CubeCL launch binding.
    pub fn binding(&self) -> ComplexTensorBinding<'_> {
        ComplexTensorBinding { handle: self }
    }

    /// Returns the underlying scalar tensor metadata and allocation.
    pub fn into_raw_parts(self) -> TensorHandle {
        self.tensor
    }
}

/// A borrowed C32 tensor handle that can produce a CubeCL tensor binding at launch time.
pub struct ComplexTensorBinding<'a> {
    handle: &'a ComplexTensorHandle,
}

impl ComplexTensorBinding<'_> {
    pub(crate) fn shape(&self) -> &[usize] {
        self.handle.shape()
    }

    pub(crate) fn strides(&self) -> &[usize] {
        self.handle.strides()
    }

    /// Whether two bindings borrow the same wrapper; allocation aliasing is
    /// checked separately by [`Self::ensure_unique_output`].
    pub(crate) fn is_same_tensor(&self, other: &Self) -> bool {
        core::ptr::eq(self.handle, other.handle)
    }

    pub(crate) fn tensor(&self) -> TensorBinding {
        self.handle.tensor.clone().binding()
    }

    pub(crate) fn ensure_unique_output(&self) -> Result<(), FftError> {
        ensure_unique_output(&self.handle.tensor)
    }
}

pub(crate) fn ensure_unique_output(tensor: &TensorHandle) -> Result<(), FftError> {
    if tensor.can_mut() {
        Ok(())
    } else {
        Err(FftError::OverlappingBindings)
    }
}

fn ensure_c32_dtype(dtype: ElemType) -> Result<(), FftError> {
    if dtype == f32::elem_type_native() {
        Ok(())
    } else {
        Err(FftError::UnsupportedDtype { actual: dtype })
    }
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, FftError> {
    if shape.contains(&0) {
        return Ok(vec![0; shape.len()]);
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (axis, extent) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride.checked_mul(*extent).ok_or(FftError::SizeOverflow)?;
    }
    Ok(strides)
}

fn scalar_layout(
    shape: &[usize],
    logical_strides: &[usize],
) -> Result<(Vec<usize>, usize), FftError> {
    let scalar_strides = logical_strides
        .iter()
        .enumerate()
        .map(|(axis, stride)| {
            stride
                .checked_mul(2)
                .ok_or(FftError::StrideOverflow { axis })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if shape.contains(&0) {
        return Ok((scalar_strides, 0));
    }

    let last_imaginary_scalar =
        shape
            .iter()
            .zip(&scalar_strides)
            .try_fold(1usize, |offset, (extent, stride)| {
                let axis_offset = (extent - 1)
                    .checked_mul(*stride)
                    .ok_or(FftError::SizeOverflow)?;
                offset
                    .checked_add(axis_offset)
                    .ok_or(FftError::SizeOverflow)
            })?;
    let physical_scalar_len = last_imaginary_scalar
        .checked_add(1)
        .ok_or(FftError::SizeOverflow)?;

    Ok((scalar_strides, physical_scalar_len))
}

#[cfg(test)]
mod tests {
    use cubecl::prelude::Scalar;

    use super::*;

    #[test]
    fn output_binding_rejects_an_aliased_allocation_before_binding() {
        let client = cubecl::test_device().client();
        let dtype = f32::elem_type_native();

        let handle = client.empty(4 * dtype.size());
        let input = handle.clone();
        let aliased = ComplexTensorHandle::new_contiguous(vec![2], handle, dtype).unwrap();
        assert!(matches!(
            aliased.binding().ensure_unique_output(),
            Err(FftError::OverlappingBindings)
        ));
        assert_eq!(input.size_in_used(), 4 * dtype.size() as u64);
    }

    #[test]
    fn invalid_handle_offset_range_returns_an_error_without_panicking() {
        let client = cubecl::test_device().client();
        let dtype = f32::elem_type_native();
        let handle = client
            .empty(4 * dtype.size())
            .offset_start(12)
            .offset_end(8);

        let result = ComplexTensorHandle::new_contiguous(vec![1], handle, dtype);

        assert!(matches!(result, Err(FftError::InvalidBufferRange { .. })));
    }
}
