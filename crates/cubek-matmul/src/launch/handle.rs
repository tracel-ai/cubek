use cubecl::quant::scheme::{BlockSize, QuantLevel};
use cubecl::std::tensor::{into_contiguous_packed, into_contiguous_pitched};
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{AddressType, StorageType},
    prelude::{CubePrimitive, TensorBinding},
    server::LaunchError,
    zspace::Shape,
};
use cubecl_common::quant::scheme::{QuantScheme, QuantStore, QuantValue};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MatmulInputBinding<R: Runtime> {
    Normal(TensorBinding<R>, StorageType),
    Quantized {
        data: TensorBinding<R>,
        data_dtype: StorageType,
        scale: TensorBinding<R>,
        scale_dtype: StorageType,
        /// Unpacked shape, excluding padding
        shape: Shape,
        scheme: QuantScheme,
    },
}

impl<R: Runtime> Clone for MatmulInputBinding<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Normal(arg0, arg1) => Self::Normal(arg0.clone_unchecked(), *arg1),
            Self::Quantized {
                data,
                data_dtype,
                scale,
                scale_dtype,
                shape,
                scheme,
            } => Self::Quantized {
                data: data.clone_unchecked(),
                data_dtype: *data_dtype,
                scale: scale.clone_unchecked(),
                scale_dtype: *scale_dtype,
                shape: shape.clone(),
                scheme: *scheme,
            },
        }
    }
}

impl<R: Runtime> MatmulInputBinding<R> {
    pub fn new(data: TensorBinding<R>, dtype: StorageType) -> Self {
        Self::Normal(data, dtype)
    }

    pub fn swap_dims(&mut self, dim0: usize, dim1: usize) {
        match self {
            Self::Normal(handle, _dtype) => {
                handle.shape.swap(dim0, dim1);
                handle.strides.swap(dim0, dim1);
            }
            Self::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype: _,
                scale_dtype: _,
            } => {
                let rank = data.shape.len();

                data.shape.swap(dim0, dim1);
                data.strides.swap(dim0, dim1);

                // Swap dims for scale and block size if block scaled quant is used
                if let QuantLevel::Block(block) = &mut scheme.level {
                    scale.shape.swap(dim0, dim1);
                    scale.strides.swap(dim0, dim1);

                    let mut block_size = block.to_dim_vec(rank);
                    block_size.swap(dim0, dim1);
                    *block = BlockSize::new_trim(block_size)
                }

                shape.swap(dim0, dim1);

                // Swap packed dim if packed dim is either of `dim0` or `dim1`
                if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
                    &mut scheme.store
                {
                    if *packed_dim == rank - dim0 - 1 {
                        *packed_dim = rank - dim1 - 1;
                    } else if *packed_dim == rank - dim1 - 1 {
                        *packed_dim = rank - dim0 - 1;
                    }
                }
            }
        }
    }
    pub fn quantized(
        data: TensorBinding<R>,
        scale: TensorBinding<R>,
        shape: Shape,
        scheme: QuantScheme,
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

    pub fn data(&self) -> &TensorBinding<R> {
        match self {
            MatmulInputBinding::Normal(handle, ..) => handle,
            MatmulInputBinding::Quantized { data, .. } => data,
        }
    }

    pub fn into_data(self) -> TensorBinding<R> {
        match self {
            MatmulInputBinding::Normal(handle, ..) => handle,
            MatmulInputBinding::Quantized { data, .. } => data,
        }
    }

    pub fn data_mut(&mut self) -> &mut TensorBinding<R> {
        match self {
            MatmulInputBinding::Normal(handle, ..) => handle,
            MatmulInputBinding::Quantized { data, .. } => data,
        }
    }

    pub fn scale(&self) -> Option<&TensorBinding<R>> {
        match self {
            MatmulInputBinding::Normal(..) => None,
            MatmulInputBinding::Quantized { scale, .. } => Some(scale),
        }
    }

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulInputBinding::Normal(..) => None,
            MatmulInputBinding::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn shape(&self) -> &Shape {
        match self {
            MatmulInputBinding::Normal(handle, ..) => &handle.shape,
            MatmulInputBinding::Quantized { shape, .. } => shape,
        }
    }

    pub fn into_contiguous(
        mut self,
        client: &ComputeClient<R>,
        ohs: &mut Self,
    ) -> Result<Self, LaunchError> {
        if self.data().handle.id == ohs.data().handle.id {
            let data = self.data_mut();
            let last_use = data.handle.last_use;
            data.handle.last_use = false;

            if last_use {
                ohs.data_mut().handle.last_use = true;
            }
        }

        let val = match self {
            Self::Normal(data, dtype) => Self::Normal(
                into_contiguous_pitched(client, data, dtype).binding(),
                dtype,
            ),
            Self::Quantized {
                data,
                scale,
                shape,
                scheme,
                data_dtype,
                scale_dtype,
            } => {
                let mut scheme = scheme;
                let data = match scheme.store {
                    // e2m1 has native packing (e2m1x2) so also needs to be re-packed
                    QuantStore::PackedNative(packed_dim) if scheme.value == QuantValue::E2M1 => {
                        let mut data = into_contiguous_packed(
                            client,
                            data,
                            packed_dim,
                            &shape,
                            scheme.num_quants(),
                            u8::as_type_native_unchecked(),
                        );
                        scheme = scheme.with_store(QuantStore::PackedNative(0));
                        data.dtype = data_dtype;
                        data
                    }
                    QuantStore::PackedU32(packed_dim) => {
                        let mut data = into_contiguous_packed(
                            client,
                            data,
                            packed_dim,
                            &shape,
                            scheme.num_quants(),
                            u32::as_type_native_unchecked(),
                        );
                        data.dtype = data_dtype;
                        scheme = scheme.with_store(QuantStore::PackedU32(0));
                        data
                    }
                    _ => into_contiguous_pitched(client, data, data_dtype),
                };

                Self::Quantized {
                    data: data.binding(),
                    scale,
                    shape,
                    scheme,
                    data_dtype,
                    scale_dtype,
                }
            }
        };

        Ok(val)
    }

    pub fn required_address_type(&self) -> AddressType {
        match self {
            MatmulInputBinding::Normal(handle, ..) => handle.required_address_type(),
            MatmulInputBinding::Quantized { data, shape, .. } => {
                let handle_addr = data.required_address_type();
                let conceptual_addr = AddressType::from_len(shape.iter().product());
                handle_addr.max(conceptual_addr)
            }
        }
    }
}
