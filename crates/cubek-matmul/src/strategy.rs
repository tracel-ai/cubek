use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};
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
    pub(crate) fn launch_ref<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs: InputBinding<R>,
        rhs: InputBinding<R>,
        out: TensorBinding<R>,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        match self {
            #[cfg(feature = "tiled")]
            Strategy::Tiled(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            #[cfg(feature = "multi-level")]
            Strategy::MultiLevel(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            Strategy::Auto => auto(client, lhs, rhs, out, dtypes),
        }
    }
}

/// Accelerated first, falling back to the routine that needs no accelerator.
#[cfg(feature = "multi-level")]
fn auto<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    if let Err(err) = multi_level::Strategy::SimpleCyclicCmma(Default::default()).launch_ref(
        client,
        lhs.clone(),
        rhs.clone(),
        out.clone(),
        dtypes,
    ) {
        match err {
            MatmulSetupError::Unavailable(_) => {
                multi_level::Strategy::SimpleUnit(Default::default())
                    .launch_ref(client, lhs, rhs, out, dtypes)?;
            }
            _ => panic!("{err:?}"),
        }
    }

    Ok(())
}

/// The tiled-only pair. `CpuGemm` takes the fallback slot as the one tiled routine with no
/// hardware requirement; it was tuned for CPU and has never been measured as a GPU fallback.
#[cfg(all(feature = "tiled", not(feature = "multi-level")))]
fn auto<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    if let Err(err) = tiled::Strategy::Cmma(Default::default()).launch_ref(
        client,
        lhs.clone(),
        rhs.clone(),
        out.clone(),
        dtypes,
    ) {
        match err {
            MatmulSetupError::Unavailable(_) => {
                tiled::Strategy::CpuGemm(Default::default())
                    .launch_ref(client, lhs, rhs, out, dtypes)?;
            }
            _ => panic!("{err:?}"),
        }
    }

    Ok(())
}
