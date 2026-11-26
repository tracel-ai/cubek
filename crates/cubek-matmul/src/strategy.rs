use std::fmt::Display;

use cubecl::{client::Client, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::definition::{MatmulElems, MatmulSetupError};
#[cfg(feature = "multi-level")]
use crate::multi_level;
#[cfg(feature = "tiled")]
use crate::tiled;

/// How to solve a matmul. The two arms are the two kernel architectures: routines written
/// on the tile DSL, and routines written on the batch/global/stage/tile levels.
#[derive(Clone, Default)]
pub enum Strategy {
    #[cfg(feature = "tiled")]
    Tiled(tiled::Strategy),
    #[cfg(feature = "multi-level")]
    MultiLevel(multi_level::Strategy),
    #[default]
    Auto,
}

#[cfg(feature = "tiled")]
impl From<tiled::Strategy> for Strategy {
    fn from(s: tiled::Strategy) -> Self {
        Strategy::Tiled(s)
    }
}

#[cfg(feature = "multi-level")]
impl From<multi_level::Strategy> for Strategy {
    fn from(s: multi_level::Strategy) -> Self {
        Strategy::MultiLevel(s)
    }
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "tiled")]
            Strategy::Tiled(s) => write!(f, "{}", s),
            #[cfg(feature = "multi-level")]
            Strategy::MultiLevel(s) => write!(f, "{}", s),
            Strategy::Auto => f.write_str("matmul_auto"),
        }
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    pub(crate) fn launch_ref(
        &self,
        client: &Client,
        lhs: InputBinding,
        rhs: InputBinding,
        out: TensorBinding,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        match self {
            #[cfg(feature = "tiled")]
            Strategy::Tiled(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            #[cfg(feature = "multi-level")]
            Strategy::MultiLevel(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            Strategy::Auto => Self::launch_auto(client, lhs, rhs, out, dtypes),
        }
    }

    /// Accelerated first, falling back to the routine that needs no accelerator. A storage-tiled
    /// operand is a fact of the data, not a knob: it was stored in tiles for the tiled cmma
    /// routine, which stages to its tiles, so it goes there (the multi-level routines refuse it).
    #[cfg(feature = "multi-level")]
    fn launch_auto(
        client: &Client,
        lhs: InputBinding,
        rhs: InputBinding,
        out: TensorBinding,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        #[cfg(feature = "tiled")]
        if lhs.data().tiling.is_tiled() || rhs.data().tiling.is_tiled() {
            return tiled::Strategy::Cmma(crate::kernels::CubeStagedMatmulRoutineSettings::DEFAULT)
                .launch_ref(client, lhs, rhs, out, dtypes);
        }
        let (first_lhs, first_rhs, first_out) = (lhs.clone(), rhs.clone(), out.clone());
        launch_with_fallback(
            dtypes,
            move |dtypes| {
                multi_level::Strategy::SimpleCyclicCmma(Default::default())
                    .launch_ref(client, first_lhs, first_rhs, first_out, dtypes)
            },
            move |dtypes| {
                multi_level::Strategy::SimpleUnit(Default::default())
                    .launch_ref(client, lhs, rhs, out, dtypes)
            },
        )
    }

    /// The tiled-only pair. `CpuGemm` takes the fallback slot as the one tiled routine with no
    /// hardware requirement; it was tuned for CPU and has never been measured as a GPU fallback.
    #[cfg(all(feature = "tiled", not(feature = "multi-level")))]
    fn launch_auto(
        client: &Client,
        lhs: InputBinding,
        rhs: InputBinding,
        out: TensorBinding,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        let (first_lhs, first_rhs, first_out) = (lhs.clone(), rhs.clone(), out.clone());
        launch_with_fallback(
            dtypes,
            move |dtypes| {
                tiled::Strategy::Cmma(crate::kernels::CubeStagedMatmulRoutineSettings::DEFAULT)
                    .launch_ref(client, first_lhs, first_rhs, first_out, dtypes)
            },
            move |dtypes| {
                tiled::Strategy::CpuGemm(
                    crate::kernels::UnitRegistersMatmulRoutineSettings::DEFAULT,
                )
                .launch_ref(client, lhs, rhs, out, dtypes)
            },
        )
    }
}

/// `first`, or `fallback` where `first` refuses the problem or the device
/// ([`MatmulSetupError::Unavailable`], [`MatmulSetupError::InvalidConfig`]), the fallback run on the
/// `dtypes` the caller passed, whatever `first` made of them. Any other error is `first`'s answer,
/// a broken launch among them.
#[allow(clippy::result_large_err)]
fn launch_with_fallback(
    dtypes: &mut MatmulElems,
    first: impl FnOnce(&mut MatmulElems) -> Result<(), MatmulSetupError>,
    fallback: impl FnOnce(&mut MatmulElems) -> Result<(), MatmulSetupError>,
) -> Result<(), MatmulSetupError> {
    let passed = dtypes.clone();
    let first = first(dtypes);
    if !matches!(
        first,
        Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_))
    ) {
        return first;
    }
    *dtypes = passed;
    fallback(dtypes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::MatmulAvailabilityError;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// What `launch_with_fallback` answers where `first` changes the element types and answers
    /// `first`, and the fallback answers `Ok` where it is handed the element types the caller
    /// passed; and whether the fallback ran.
    fn launched(first: MatmulSetupError) -> (Result<(), MatmulSetupError>, bool) {
        let mut dtypes = MatmulElems::from_single_dtype(F32);
        let mut fell_back = false;
        let launched = launch_with_fallback(
            &mut dtypes,
            |dtypes| {
                *dtypes = MatmulElems::from_single_dtype(F16);
                Err(first)
            },
            |dtypes| {
                fell_back = true;
                match *dtypes == MatmulElems::from_single_dtype(F32) {
                    true => Ok(()),
                    false => Err(MatmulSetupError::InvalidConfig(Box::new("changed dtypes"))),
                }
            },
        );
        (launched, fell_back)
    }

    #[test]
    fn a_feature_the_device_lacks_falls_back_on_the_dtypes_passed() {
        let (launched, fell_back) = launched(MatmulAvailabilityError::PlaneOpsUnavailable.into());

        assert!(fell_back);
        assert!(launched.is_ok());
    }

    #[test]
    fn a_refused_problem_falls_back() {
        let (launched, fell_back) = launched(MatmulSetupError::InvalidConfig(Box::new("refused")));

        assert!(fell_back);
        assert!(launched.is_ok());
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn a_broken_launch_does_not_fall_back() {
        let broken = cubecl::tile::launch::Decline::Broken {
            condition: "the routine holds",
            cause: None,
        };

        let (launched, fell_back) = launched(broken.clone().into());

        assert!(!fell_back);
        assert!(matches!(launched, Err(MatmulSetupError::Broken(decline)) if decline == broken));
    }
}
