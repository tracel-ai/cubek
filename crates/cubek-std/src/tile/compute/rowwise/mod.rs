//! Row-wise tile compute. The `Tile<E, Plane, ReadWrite>` impl in [`dispatch`]
//! routes per-variant `row_max`, `row_sum`, `exp_diff`, and `rowwise_scale`
//! to the right backend; [`reducer`] handles the cross-unit plane reduction
//! used by the `WhiteboxFragment` and `BounceTile` arms.

mod dispatch;
pub(crate) mod reducer;
