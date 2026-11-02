use crate::attention::{DV, Q, S};
use crate::attention_backward::key_gradient::plane::PlaneKeyGradientMicrokernel;
use crate::attention_backward::{ScoreWindows, scale_rows};
use crate::matmul::MatmulOperands;
use crate::matmul::plane::{
    PlaneFragmentDrain, PlaneFragmentMicrokernel, plane_fragment_microkernel,
};
use cubecl::tile::kind::PlanesOutput;
use cubecl::tile::{Level, Region, Scratch, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// What a walk owned by keys adds into the query's gradient beside its own gradients:
/// `scale · score_gradient · key` for each stage's queries, into `sum`, a sum every cube that sees
/// those queries adds into.
///
/// The cube's planes split its keys. A cube reading its score through windows has each plane
/// contract its own keys for every query of the stage, and the planes meet in shared memory
/// ([`StageQueryGradientShare`]). A cube holding its score in fragments leaves the score's gradient
/// of every plane in one window, and each plane contracts all the cube's keys for its own rows of
/// the stage's queries, so no two planes add into the same rows ([`add_rows`](Self::add_rows)).
#[derive(CubeType, Clone)]
pub struct QueryGradientShare {
    pub sum: Tile,
    /// The product's fragments: a stage's queries by the whole head dim, walking the plane's keys;
    /// or, by rows, a plane's rows of them, walking the cube's keys.
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
    /// `plane`'s rows of `stage`'s query gradient, `planes` planes sharing the stage's `rows`
    /// queries: `score_gradient · key` contracted over all the cube's keys, `score_gradient` the
    /// window every plane left its part of the score's gradient in and `key` the cube's staged
    /// keys, scaled by `scale` and added into the sum. The caller meets the cube before, once every
    /// plane has left its part, and after, before any plane writes the window again.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rows<EA: Float>(
        &self,
        stage: &Region,
        plane: &Region,
        score_gradient: &Tile,
        key: &Tile,
        scale: EA,
        #[comptime] planes: usize,
        #[comptime] rows: usize,
    ) {
        let share = comptime!(rows / planes);
        let own = comptime!(Level::every(&[(Q, share)]));
        let at = plane.coord(S);
        let sum = self.sum.at(stage);
        let sum = sum.at(&sum.over(&own).region(at));
        let score_gradient = score_gradient.at(&score_gradient.over(&own).region(at));
        // The part is the plane's own, on the product's grid with no planes above it.
        let product = comptime!(self.product);
        let part = ScoreWindows::contract_held::<EA>(
            &score_gradient,
            key,
            comptime!(sum.space()),
            product,
            comptime!(PlaneKeyGradientMicrokernel::grid_levels(product, &[])),
        );
        scale_rows::<EA>(&part, scale, share);
        ScoreWindows::store_cast(&sum, &part, product);
    }

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
    /// `scale`, drained into the meeting. Every plane of the cube adds its part. The part bounces
    /// through `scratch` as it is scaled where it drains by bouncing, a cmma one.
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
        let scratch = comptime!(match self.product.instruction.drain() {
            PlaneFragmentDrain::Bounce => scratch,
            PlaneFragmentDrain::CellByCell | PlaneFragmentDrain::InLines => Scratch::None,
        });
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
