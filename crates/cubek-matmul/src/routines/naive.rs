use cubecl::client::ComputeClient;

use crate::{
    components::batch::{BatchMatmulFamily, naive::NaiveBatchMatmulFamily},
    definition::{MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError},
    routines::Routine,
};

pub struct NaiveRoutine {}

impl Routine for NaiveRoutine {
    type Strategy = ();
    type Blueprint = ();

    type BatchMatmul = NaiveBatchMatmulFamily;
    type Config = <Self::BatchMatmul as BatchMatmulFamily>::Config;

    fn prepare<R: cubecl::Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        plane_dim: u32,
        line_sizes: &MatmulLineSizes,
        args: &Self::Strategy,
        dtypes: &mut MatmulElems,
    ) -> Result<Self::Blueprint, MatmulSetupError> {
        todo!()
    }

    fn can_cast_stage_element() -> bool {
        // Irrelevant
        false
    }
}
