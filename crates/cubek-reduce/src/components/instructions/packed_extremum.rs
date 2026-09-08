use cubecl::prelude::*;

use crate::components::{
    instructions::{Accumulator, Item, Packed, Packing, ReduceStep, Value},
    precision::ReducePrecision,
};

/// Max and min accumulated as one packed candidate, where the value's rank and the
/// coordinate tie-break ride a single unsigned comparison.
///
/// The two extremes share every step but the [`ValueOrder`] it is packed in
/// and the identity it starts from. In particular both reduce a plane with
/// `plane_max`: the packing ranks either extreme's winner highest, so min does not
/// want `plane_min` here.
#[derive(Debug, CubeType, Clone)]
pub struct PackedExtremum {
    packing: Packing,
}

#[cube]
impl PackedExtremum {
    /// Ranks the largest value first, as max wants.
    pub fn descending() -> PackedExtremum {
        PackedExtremum {
            packing: Packing::descending(),
        }
    }

    /// Ranks the smallest value first, as min wants.
    pub fn ascending() -> PackedExtremum {
        PackedExtremum {
            packing: Packing::ascending(),
        }
    }

    pub fn null_accumulator<P: ReducePrecision>(&self, identity: P::EA) -> Accumulator<P> {
        Accumulator::new_Packed(Value::new_single(
            self.packing.empty::<P::EA, P::SI>(Vector::new(identity)),
        ))
    }

    pub fn reduce<P: ReducePrecision>(
        &self,
        packed: &mut Value<Vector<Packed, P::SI>>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let candidate = self
            .packing
            .pack::<P::EA, P::SI>(Vector::cast_from(item.elements), item.args.item());

        let candidate = match reduce_step {
            ReduceStep::Plane => plane_max(candidate),
            ReduceStep::Identity => candidate,
        };

        Packing::insert::<P::SI>(packed, candidate);
    }

    pub fn plane_reduce_inplace<P: ReducePrecision>(
        &self,
        packed: &mut Value<Vector<Packed, P::SI>>,
    ) {
        let winning = plane_max(packed.item());
        packed.assign(&Value::new_single(winning));
    }

    pub fn fuse_accumulators<P: ReducePrecision>(
        &self,
        packed: &mut Value<Vector<Packed, P::SI>>,
        other: Vector<Packed, P::SI>,
    ) {
        Packing::insert::<P::SI>(packed, other);
    }

    pub fn to_output_parallel<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        packed: Vector<Packed, P::SI>,
    ) -> (Value<Out>, Value<Idx>) {
        let candidate = Vector::<Packed, Const<1>>::new(Packing::finalize::<P::SI>(packed));

        (
            Value::new_single(Out::cast_from(
                self.packing
                    .value::<P::EA, Const<1>>(candidate)
                    .extract(0usize),
            )),
            Value::new_single(Idx::cast_from(
                Packing::coordinate::<Const<1>>(candidate).extract(0usize),
            )),
        )
    }

    pub fn to_output_perpendicular<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        candidate: Vector<Packed, P::SI>,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(
                self.packing.value::<P::EA, P::SI>(candidate),
            )),
            Value::new_single(Vector::cast_from(Packing::coordinate::<P::SI>(candidate))),
        )
    }
}
