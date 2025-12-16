use cubecl::client::ComputeClient;

use crate::{
    components::batch::{
        BatchMatmulFamily,
        naive::{NaiveBatchMatmulFamily, NaiveBlueprint},
    },
    definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError},
    routines::Routine,
};

pub struct NaiveRoutine {}

impl Routine for NaiveRoutine {
    type Strategy = ();
    type BatchMatmul = NaiveBatchMatmulFamily;
    type Blueprint = <Self::BatchMatmul as BatchMatmulFamily>::Blueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: cubecl::Runtime>(
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        _plane_dim: u32,
        _line_sizes: &MatmulLineSizes,
        _args: &Self::Strategy,
        _dtypes: &mut MatmulElems,
    ) -> Result<Self::Blueprint, MatmulSetupError> {
        Ok(NaiveBlueprint {})
    }

    fn can_cast_stage_element() -> bool {
        // Irrelevant
        false
    }
}
