use cubecl::prelude::*;

#[derive(Debug, Clone, Copy)]
/// Matrix dimension specifier for matmul operations.
pub enum MatmulDim {
    /// Rows of the output matrix.
    M,
    /// Columns of the output matrix.
    N,
    /// Reduction dimension.
    K,
}

#[macro_export]
macro_rules! define_3d_size_base {
    ($name:ident, $ty:ty) => {
        #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
        pub struct $name {
            pub m: $ty,
            pub n: $ty,
            pub k: $ty,
        }

        impl $name {
            pub fn new(m: u32, n: u32, k: u32) -> Self {
                $name {
                    m: <$ty>::try_from(m).unwrap(),
                    n: <$ty>::try_from(n).unwrap(),
                    k: <$ty>::try_from(k).unwrap(),
                }
            }

            pub fn get(&self, dim: $crate::MatmulDim) -> u32 {
                (match dim {
                    $crate::MatmulDim::M => self.m,
                    $crate::MatmulDim::N => self.n,
                    $crate::MatmulDim::K => self.k,
                }) as u32
            }

            pub fn m(&self) -> u32 {
                self.get($crate::MatmulDim::M)
            }

            pub fn n(&self) -> u32 {
                self.get($crate::MatmulDim::N)
            }

            pub fn k(&self) -> u32 {
                self.get($crate::MatmulDim::K)
            }

            pub fn mn(&self) -> u32 {
                self.get($crate::MatmulDim::M) * self.get($crate::MatmulDim::N)
            }

            pub fn mk(&self) -> u32 {
                self.get($crate::MatmulDim::M) * self.get($crate::MatmulDim::K)
            }

            pub fn nk(&self) -> u32 {
                self.get($crate::MatmulDim::N) * self.get($crate::MatmulDim::K)
            }

            pub fn mnk(&self) -> u32 {
                self.get($crate::MatmulDim::M)
                    * self.get($crate::MatmulDim::N)
                    * self.get($crate::MatmulDim::K)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_3d_size_from_tuple {
    ($name:ident, $ty_struct:ty, $ty_tuple:ty) => {
        impl From<($ty_tuple, $ty_tuple, $ty_tuple)> for $name {
            fn from(value: ($ty_tuple, $ty_tuple, $ty_tuple)) -> Self {
                Self {
                    m: value.0 as $ty_struct,
                    n: value.1 as $ty_struct,
                    k: value.2 as $ty_struct,
                }
            }
        }

        impl From<$name> for ($ty_tuple, $ty_tuple, $ty_tuple) {
            fn from(value: $name) -> Self {
                (
                    value.m as $ty_tuple,
                    value.n as $ty_tuple,
                    value.k as $ty_tuple,
                )
            }
        }
    };
}

// Shapes m,n,k of the problem
define_3d_size_base!(MatmulProblemSize, u32);
impl_3d_size_from_tuple!(MatmulProblemSize, u32, u8);
impl_3d_size_from_tuple!(MatmulProblemSize, u32, u32);
impl_3d_size_from_tuple!(MatmulProblemSize, u32, i32);
impl_3d_size_from_tuple!(MatmulProblemSize, u32, u16);
impl_3d_size_from_tuple!(MatmulProblemSize, u32, usize);
