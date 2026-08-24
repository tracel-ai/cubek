use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    multi_level, tiled,
};

/// How to solve a matmul. The two arms are the two kernel families: routines written on
/// the tile DSL, and routines written on the batch/global/stage/tile levels.
#[derive(Clone, Default)]
pub enum Strategy {
    Tiled(tiled::Strategy),
    MultiLevel(multi_level::Strategy),
    #[default]
    Auto,
}

impl From<tiled::Strategy> for Strategy {
    fn from(s: tiled::Strategy) -> Self {
        Strategy::Tiled(s)
    }
}

impl From<multi_level::Strategy> for Strategy {
    fn from(s: multi_level::Strategy) -> Self {
        Strategy::MultiLevel(s)
    }
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Tiled(s) => write!(f, "{}", s),
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
            Strategy::Tiled(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            Strategy::MultiLevel(s) => s.launch_ref(client, lhs, rhs, out, dtypes),
            Strategy::Auto => auto(client, lhs, rhs, out, dtypes),
        }
    }
}

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
