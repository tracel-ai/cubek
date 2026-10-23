//! The two tile-DSL routines a caller of [`launch_ref`](crate::launch::launch_ref) names: settings
//! of the dense routines in [`kernels`](crate::kernels).

use std::fmt::Display;

use cubecl::{client::Client, prelude::TensorBinding};
use cubek_std::{Hardware, InputBinding};

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    kernels::{
        CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulRoutine,
        UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings, tensor::PlainTensor,
    },
};

#[derive(Clone)]
pub enum Strategy {
    /// [`CubeStagedMatmulRoutine`] at these settings: a cube's planes contract a stage the cube
    /// fills, on the matrix instruction the device offers.
    Cmma(CubeStagedMatmulRoutineSettings),
    /// [`UnitRegistersMatmulRoutine`] at these settings: every unit holds a block of the output in
    /// its registers. Needs no accelerator.
    CpuGemm(UnitRegistersMatmulRoutineSettings),
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Cmma(settings) => write!(f, "cmma {settings}"),
            Strategy::CpuGemm(settings) => write!(f, "cpu_gemm {settings}"),
        }
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    /// `lhs · rhs` written into `out`, all three of one float element: the routines read no
    /// quantized operand, and contract in the operands' element, summing in `f32`.
    pub(crate) fn launch_ref(
        &self,
        client: &Client,
        lhs: InputBinding,
        rhs: InputBinding,
        out: TensorBinding,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        let (InputBinding::Normal(lhs, lhs_elem), InputBinding::Normal(rhs, rhs_elem)) = (lhs, rhs)
        else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "{self} reads no quantized operand"
            ))));
        };
        if dtypes.acc_global != lhs_elem {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "{self} writes the operands' element {lhs_elem:?}, not {:?}",
                dtypes.acc_global
            ))));
        }
        let hw = Hardware::from_client(client);
        let (lhs, rhs) = (
            PlainTensor::new(lhs, lhs_elem),
            PlainTensor::new(rhs, rhs_elem),
        );
        let out = PlainTensor::new(out, lhs_elem);
        match self {
            Strategy::Cmma(settings) => {
                CubeStagedMatmulRoutine.launch_into(&hw, client, lhs, rhs, out, settings)
            }
            Strategy::CpuGemm(settings) => {
                UnitRegistersMatmulRoutine.launch_into(&hw, client, lhs, rhs, out, settings)
            }
        }
        .map_err(MatmulSetupError::from)
    }
}
