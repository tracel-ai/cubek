use crate::{components::batch::naive::NaiveBatchMatmulFamily, routines::Routine};

pub struct Naive {}

impl Routine for Naive {
    type SelectionArgs;

    type TileMatmul;

    type StageMatmul;

    type GlobalMatmul;

    type BatchMatmul = NaiveBatchMatmulFamily;

    fn selection<R: cubecl::Runtime>(
        client: &cubecl::prelude::ComputeClient<R>,
        problem: &crate::definition::MatmulProblem,
        plane_dim: u32,
        line_sizes: &crate::definition::MatmulLineSizes,
        args: &Self::SelectionArgs,
        dtypes: &mut crate::definition::MatmulElems,
    ) -> Result<crate::definition::MatmulSelection, crate::definition::MatmulSetupError> {
        todo!()
    }
}
