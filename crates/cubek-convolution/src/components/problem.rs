use cubecl::{
    ir::AddressType,
    zspace::{Shape, Strides, shape},
};
use cubek_matmul::definition::{MatmulGlobalElems, MatmulProblem};
use cubek_std::MatrixLayout;

#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConvolutionOperation {
    Forward,
    BackwardData,
    BackwardWeight,
    ForwardTransposed,
}

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct ConvolutionProblem {
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub lhs_strides: Strides,
    pub rhs_strides: Strides,

    pub lhs_layout: MatrixLayout,
    pub rhs_layout: MatrixLayout,

    pub kernel_size: Vec<u32>,
    pub stride: Vec<u32>,
    pub padding: Vec<i32>,
    pub dilation: Vec<u32>,

    pub batches: usize,
    pub channels: usize,
    pub out_channels: usize,
    pub in_shape: Shape,
    pub out_shape: Shape,

    /// Channels after applying loader-specific padding
    pub padded_channels: usize,
    pub operation: ConvolutionOperation,

    pub dimensionality: Dimensionality,

    pub global_dtypes: MatmulGlobalElems,
    /// Address type, defined as the max of each handle's `required_address_type`
    pub address_type: AddressType,
}

impl ConvolutionProblem {
    pub fn as_matmul_problem(&self) -> MatmulProblem {
        let rank = self.lhs_strides.len();

        // Strides are expected to be in row major (m, n) format so for matmul checks we need to
        // convert them to that format, with all other dims treated as batch dims so they're still
        // checked.
        // lhs already has the right format, but rhs needs special handling.
        // (h, w, c, n)
        let lhs_strides = match self.lhs_layout {
            MatrixLayout::RowMajor => self.lhs_strides.clone(),
            MatrixLayout::ColMajor => {
                let mut lhs_strides: Strides = self.lhs_strides[1..rank].into();
                lhs_strides.push(self.lhs_strides[0]);
                lhs_strides
            }
        };
        let rhs_strides = match self.rhs_layout {
            MatrixLayout::RowMajor => self.rhs_strides.clone(),
            MatrixLayout::ColMajor => {
                let mut rhs_strides: Strides = self.rhs_strides[1..rank].into();
                rhs_strides.push(self.rhs_strides[0]);
                rhs_strides
            }
        };

        MatmulProblem {
            m: self.m,
            n: self.n,
            k: self.k,
            lhs_batches: shape![],
            rhs_batches: shape![],
            out_batches: shape![],
            lhs_strides,
            rhs_strides,
            lhs_layout: self.lhs_layout,
            rhs_layout: self.rhs_layout,
            lhs_shape: shape![self.m, self.k],
            rhs_shape: shape![self.k, self.n],
            out_shape: shape![self.m, self.n],
            out_strides: MatrixLayout::RowMajor.to_strides(&[self.m, self.n]),
            out_layout: MatrixLayout::RowMajor,
            lhs_scheme: None,
            rhs_scheme: None,
            global_dtypes: self.global_dtypes.clone(),
            address_type: self.address_type,
        }
    }

    pub fn should_check_channel(&self) -> bool {
        self.channels != self.padded_channels
    }

    pub fn should_check_spatial_bounds(&self) -> bool {
        self.padding.iter().any(|&pad| pad != 0)
    }
}

/// Spatial dimensionality of an operation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Dimensionality {
    Dim1,
    Dim2,
    Dim3,
}

impl Dimensionality {
    pub fn num_dims(&self) -> usize {
        match self {
            Dimensionality::Dim1 => 1,
            Dimensionality::Dim2 => 2,
            Dimensionality::Dim3 => 3,
        }
    }
}
