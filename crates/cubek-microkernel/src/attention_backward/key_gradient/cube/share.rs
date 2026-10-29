use crate::attention::DV;
use crate::attention_backward::scale_rows;
use crate::matmul::MatmulOperands;
use crate::matmul::plane::{PlaneFragmentMicrokernel, plane_fragment_microkernel};
use cubecl::prelude::*;
use cubecl::tile::kind::PlanesOutput;
use cubecl::tile::{Region, Scratch, Semiring, Tile};
use cubecl::{self as cubecl, cube};

/// What a walk owned by keys adds into the query's gradient beside its own gradients:
/// `scale · score_gradient · key` for each stage's queries, into `sum`, a sum every cube that sees
/// those queries adds into.
#[derive(CubeType, Clone)]
pub struct QueryGradientShare {
    pub sum: Tile,
    /// The product's fragments: a stage's queries by the whole head dim, walking the plane's keys.
    #[cube(comptime)]
    pub product: PlaneFragmentMicrokernel,
}

/// One stage's share, its planes' products met in shared memory where they split the stage's keys,
/// so the cube adds into the sum once.
#[derive(CubeType)]
pub struct StageQueryGradientShare<EA: Float> {
    sum: Tile,
    output: PlanesOutput<EA>,
    #[cube(comptime)]
    product: PlaneFragmentMicrokernel,
}

#[cube]
impl QueryGradientShare {
    /// The share of `stage`, whose planes each add theirs.
    pub fn open<EA: Float>(&self, stage: &Region) -> StageQueryGradientShare<EA> {
        let sum = self.sum.at(stage);
        StageQueryGradientShare::<EA> {
            output: sum.planes_output::<EA>(stage),
            sum,
            product: comptime!(self.product),
        }
    }
}

#[cube]
impl<EA: Float> StageQueryGradientShare<EA> {
    /// `plane`'s part: `score_gradient · key` contracted over its keys, its `rows` rows scaled by
    /// `scale`, drained into the meeting. Every plane of the cube adds its part.
    pub fn add(
        &self,
        plane: &Region,
        score_gradient: &Tile,
        key: &Tile,
        scale: EA,
        #[comptime] scratch: Scratch,
        #[comptime] rows: usize,
    ) {
        let instruction = comptime!(self.product.tile_instruction());
        let mut part = self
            .sum
            .at(plane)
            .accumulator::<EA>(score_gradient, key, instruction, Semiring::SUM_PROD)
            .with_scratch(scratch);
        part.zero();
        plane_fragment_microkernel(
            plane,
            &part.clone(),
            &ComptimeOption::<Tile>::new_None(),
            &score_gradient.clone(),
            key,
            instruction,
            comptime!(Some(DV)),
            MatmulOperands::AS_BOUND,
        );

        scale_rows::<EA>(&part, scale, rows);
        self.output.drain(&part, plane);
    }

    /// The stage's sum added into the query's gradient, once every plane has added its part.
    pub fn write(&mut self) {
        self.output.write();
    }
}
