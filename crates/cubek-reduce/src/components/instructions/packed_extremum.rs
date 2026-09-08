use cubecl::prelude::*;

use crate::components::{
    instructions::{
        Accumulator, Item, OrderKey, ReduceStep, Value, ValueOrder, empty_order_key, finalize_key,
        key_insert, order_key_coordinate, order_key_value, pack_order_key,
    },
    precision::ReducePrecision,
};

/// Max and min accumulated as one packed key, where the value's rank and the
/// coordinate tie-break ride a single unsigned comparison.
///
/// The two extremes share every step but the [`ValueOrder`] the key is built in
/// and the identity it starts from. In particular both reduce a plane with
/// `plane_max`: the key ranks either extreme's winner highest, so min does not
/// want `plane_min` here.
#[derive(Debug, CubeType, Clone)]
pub struct PackedExtremum {
    #[cube(comptime)]
    pub order: ValueOrder,
}

#[cube]
impl PackedExtremum {
    /// Ranks the largest value first, as max wants.
    pub fn descending() -> PackedExtremum {
        PackedExtremum {
            order: ValueOrder::Descending,
        }
    }

    /// Ranks the smallest value first, as min wants.
    pub fn ascending() -> PackedExtremum {
        PackedExtremum {
            order: ValueOrder::Ascending,
        }
    }

    pub fn null_accumulator<P: ReducePrecision>(&self, identity: P::EA) -> Accumulator<P> {
        Accumulator::new_Packed(Value::new_single(empty_order_key::<P::EA, P::SI>(
            Vector::new(identity),
            self.order,
        )))
    }

    pub fn reduce<P: ReducePrecision>(
        &self,
        keys: &mut Value<Vector<OrderKey, P::SI>>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let key = pack_order_key::<P::EA, P::SI>(
            Vector::cast_from(item.elements),
            item.args.item(),
            self.order,
        );

        let candidate = match reduce_step {
            ReduceStep::Plane => plane_max(key),
            ReduceStep::Identity => key,
        };

        key_insert::<P::SI>(keys, candidate);
    }

    pub fn plane_reduce_inplace<P: ReducePrecision>(
        &self,
        keys: &mut Value<Vector<OrderKey, P::SI>>,
    ) {
        let winning = plane_max(keys.item());
        keys.assign(&Value::new_single(winning));
    }

    pub fn fuse_accumulators<P: ReducePrecision>(
        &self,
        keys: &mut Value<Vector<OrderKey, P::SI>>,
        other: Vector<OrderKey, P::SI>,
    ) {
        key_insert::<P::SI>(keys, other);
    }

    pub fn to_output_parallel<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        keys: Vector<OrderKey, P::SI>,
    ) -> (Value<Out>, Value<Idx>) {
        let key = Vector::<OrderKey, Const<1>>::new(finalize_key::<P::SI>(keys));

        (
            Value::new_single(Out::cast_from(
                order_key_value::<P::EA, Const<1>>(key, self.order).extract(0usize),
            )),
            Value::new_single(Idx::cast_from(
                order_key_coordinate::<Const<1>>(key).extract(0usize),
            )),
        )
    }

    pub fn to_output_perpendicular<P: ReducePrecision, Out: Numeric, Idx: Numeric>(
        &self,
        key: Vector<OrderKey, P::SI>,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        (
            Value::new_single(Vector::cast_from(order_key_value::<P::EA, P::SI>(
                key, self.order,
            ))),
            Value::new_single(Vector::cast_from(order_key_coordinate::<P::SI>(key))),
        )
    }
}
