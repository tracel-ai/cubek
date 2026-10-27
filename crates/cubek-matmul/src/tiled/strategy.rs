//! The two tile-DSL routines a caller of [`launch_ref`](crate::launch::launch_ref) names: settings
//! of the dense routines in [`kernels`](crate::kernels).

use std::fmt::Display;

use cubecl::prelude::{Scalar, tf32};
use cubecl::{client::Client, prelude::TensorBinding};
use cubek_std::config::MatmulConfig;
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

/// `dtypes` with `f32` inputs read at tf32 where the process allows it and the device takes it:
/// what the cube staged routine's planes read them at, stated as the multi-level routines state
/// their own rounding.
fn read_rounded(client: &Client, dtypes: &mut MatmulElems) {
    let (f32, tf32) = (f32::elem_type_native(), tf32::elem_type_native());
    if dtypes.lhs_global == f32
        && dtypes.rhs_global == f32
        && MatmulConfig::get().allow_tf32
        && client.properties().features.supports_type(tf32)
    {
        dtypes.lhs_stage = tf32;
        dtypes.rhs_stage = tf32;
        dtypes.lhs_register = tf32;
        dtypes.rhs_register = tf32;
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    /// `lhs · rhs` of one element written into `out`, in the element `dtypes` names for it: the
    /// routines read no quantized operand, and cast their sum to the output's element as it drains.
    /// `dtypes` comes back with the elements the planes may read the inputs at.
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
        let hw = Hardware::from_client(client);
        let (lhs, rhs) = (
            PlainTensor::new(lhs, lhs_elem),
            PlainTensor::new(rhs, rhs_elem),
        );
        let out = PlainTensor::new(out, dtypes.acc_global);
        match self {
            Strategy::Cmma(settings) => {
                read_rounded(client, dtypes);
                CubeStagedMatmulRoutine.launch_into(&hw, client, lhs, rhs, out, settings)
            }
            Strategy::CpuGemm(settings) => {
                UnitRegistersMatmulRoutine.launch_into(&hw, client, lhs, rhs, out, settings)
            }
        }
        .map_err(MatmulSetupError::from)
    }
}
