use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::StorageType,
    prelude::{CubePrimitive, TensorHandleRef},
    server::LaunchError,
};
use cubecl_common::quant::scheme::{QuantScheme, QuantStore, QuantValue};

use cubecl::std::tensor::{TensorHandle, into_contiguous_packed, into_contiguous_pitched};

pub enum MatmulInputHandle<R: Runtime> {
    Normal(TensorHandle<R>),
    Quantized {
        data: TensorHandle<R>,
        scale: TensorHandle<R>,
        shape: Vec<usize>,
        scheme: QuantScheme,
    },
}

impl<R: Runtime> MatmulInputHandle<R> {
    pub fn as_ref(&self) -> MatmulInputHandleRef<'_, R> {
        match self {
            MatmulInputHandle::Normal(handle) => {
                MatmulInputHandleRef::Normal(handle.as_ref(), handle.dtype)
            }
            MatmulInputHandle::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => MatmulInputHandleRef::Quantized {
                data: data.as_ref(),
                scale: scale.as_ref(),
                data_dtype: data.dtype,
                scale_dtype: scale.dtype,
                shape,
                scheme,
            },
        }
    }

    pub fn from_ref(handle: &MatmulInputHandleRef<'_, R>) -> Self {
        match handle {
            MatmulInputHandleRef::Normal(handle, dtype) => {
                MatmulInputHandle::Normal(TensorHandle::from_ref(handle, *dtype))
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype,
                scale_dtype,
            } => MatmulInputHandle::Quantized {
                data: TensorHandle::from_ref(data, *data_dtype),
                scale: TensorHandle::from_ref(scale, *scale_dtype),
                shape: shape.to_vec(),
                scheme: **scheme,
            },
        }
    }

    pub fn data(&self) -> &TensorHandle<R> {
        match self {
            MatmulInputHandle::Normal(handle) => handle,
            MatmulInputHandle::Quantized { data, .. } => data,
        }
    }

    pub fn swap_dims(&mut self, dim0: usize, dim1: usize) {
        match self {
            MatmulInputHandle::Normal(handle) => {
                handle.shape.swap(dim0, dim1);
                handle.strides.swap(dim0, dim1);
            }
            MatmulInputHandle::Quantized {
                data, scale, shape, ..
            } => {
                data.shape.swap(dim0, dim1);
                data.strides.swap(dim0, dim1);
                if scale.shape.len() == data.shape.len() {
                    scale.shape.swap(dim0, dim1);
                    scale.strides.swap(dim0, dim1);
                }
                shape.swap(dim0, dim1);
            }
        }
    }
}

impl<R: Runtime> Clone for MatmulInputHandle<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(handle) => Self::Normal(handle.clone()),
            Self::Quantized {
                data,
                scale,
                shape,
                scheme,
            } => Self::Quantized {
                data: data.clone(),
                scale: scale.clone(),
                shape: shape.clone(),
                scheme: *scheme,
            },
        }
    }
}

#[derive(Debug)]
pub enum MatmulInputHandleRef<'a, R: Runtime> {
    Normal(TensorHandleRef<'a, R>, StorageType),
    Quantized {
        data: TensorHandleRef<'a, R>,
        data_dtype: StorageType,
        scale: TensorHandleRef<'a, R>,
        scale_dtype: StorageType,
        /// Unpacked shape, excluding padding
        shape: &'a [usize],
        scheme: &'a QuantScheme,
    },
}

impl<'a, R: Runtime> Clone for MatmulInputHandleRef<'a, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: Runtime> Copy for MatmulInputHandleRef<'a, R> {}

impl<'a, R: Runtime> MatmulInputHandleRef<'a, R> {
    pub fn new(data: TensorHandleRef<'a, R>, dtype: StorageType) -> Self {
        Self::Normal(data, dtype)
    }

    pub fn quantized(
        data: TensorHandleRef<'a, R>,
        scale: TensorHandleRef<'a, R>,
        shape: &'a [usize],
        scheme: &'a QuantScheme,
        data_dtype: StorageType,
        scale_dtype: StorageType,
    ) -> Self {
        Self::Quantized {
            data,
            scale,
            shape,
            scheme,
            data_dtype,
            scale_dtype,
        }
    }

    pub fn data(&self) -> &TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn data_mut(&mut self) -> &mut TensorHandleRef<'a, R> {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle,
            MatmulInputHandleRef::Quantized { data, .. } => data,
        }
    }

    pub fn scale(&self) -> Option<&TensorHandleRef<'a, R>> {
        match self {
            MatmulInputHandleRef::Normal(..) => None,
            MatmulInputHandleRef::Quantized { scale, .. } => Some(scale),
        }
    }

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulInputHandleRef::Normal(..) => None,
            MatmulInputHandleRef::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            MatmulInputHandleRef::Normal(handle, ..) => handle.shape,
            MatmulInputHandleRef::Quantized { shape, .. } => shape,
        }
    }

    pub fn into_contiguous(
        &self,
        client: &ComputeClient<R>,
    ) -> Result<MatmulInputHandle<R>, LaunchError> {
        let val = match self {
            MatmulInputHandleRef::Normal(data, dtype) => {
                MatmulInputHandle::Normal(into_contiguous_pitched(client, data, *dtype)?)
            }
            MatmulInputHandleRef::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype,
                scale_dtype,
            } => {
                let data = match scheme.store {
                    // e2m1 has native packing (e2m1x2) so also needs to be re-packed
                    QuantStore::Native if scheme.value == QuantValue::E2M1 => {
                        let data = into_contiguous_packed(
                            client,
                            data,
                            shape,
                            2,
                            u8::as_type_native_unchecked(),
                        )?;
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref(), *data_dtype)
                    }
                    QuantStore::U32 => {
                        let data = into_contiguous_packed(
                            client,
                            data,
                            shape,
                            scheme.num_quants() as u32,
                            u32::as_type_native_unchecked(),
                        )?;
                        // Unsafely cast to E
                        TensorHandle::from_ref(&data.as_ref(), *data_dtype)
                    }
                    _ => into_contiguous_pitched(client, data, *data_dtype)?,
                };
                MatmulInputHandle::Quantized {
                    data,
                    scale: TensorHandle::from_ref(scale, *scale_dtype),
                    shape: shape.to_vec(),
                    scheme: **scheme,
                }
            }
        };

        Ok(val)
    }
}
