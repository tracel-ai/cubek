/// Implements `ReduceInstruction` for a public instruction by forwarding every method to the
/// crate-private core that `core` builds, so the public type keeps its name and kernel ids.
macro_rules! forward_reduce_instruction {
    ($ty:ident with indices, $($rest:tt)*) => {
        forward_reduce_instruction!($ty, $($rest)*);

        impl<P: $crate::ReducePrecision> $crate::components::instructions::ReduceWithIndices<P>
            for $ty
        {
        }
    };
    (
        $ty:ident,
        config: $config:ty,
        shared_accumulator: $shared:ty,
        from_config: |$config_arg:ident| $from_config:expr,
        core: fn $core:ident($core_this:ident: &Self) -> $core_ty:ty $core_body:block $(,)?
    ) => {
        const _: () = {
            use $crate::components::instructions::{
                Accumulator, AccumulatorFormat, Item, ReduceInstruction, ReduceOutputMode,
                ReduceRequirements, ReduceStep, Value,
            };
            use $crate::ReducePrecision;
            use cubecl::prelude::*;

            #[cube]
            impl $ty {
                fn $core($core_this: &Self) -> $core_ty $core_body
            }

            #[cube]
            impl<P: ReducePrecision> ReduceInstruction<P> for $ty {
                type SharedAccumulator = $shared;
                type Config = $config;

                fn requirements(this: &Self) -> ReduceRequirements {
                    Self::$core(this).requirements()
                }

                fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
                    Self::$core(this).accumulator_format()
                }

                fn from_config(#[comptime] $config_arg: Self::Config) -> Self {
                    $from_config
                }

                fn null_input(this: &Self) -> Vector<P::EI, P::SI> {
                    Self::$core(this).null_input::<P>()
                }

                fn null_accumulator(this: &Self) -> Accumulator<P> {
                    Self::$core(this).null_accumulator::<P>()
                }

                fn reduce(
                    this: &Self,
                    accumulator: &mut Accumulator<P>,
                    item: Item<P>,
                    #[comptime] reduce_step: ReduceStep,
                ) {
                    Self::$core(this).reduce::<P>(accumulator, item, reduce_step)
                }

                fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
                    Self::$core(this).plane_reduce_inplace::<P>(accumulator)
                }

                fn fuse_accumulators(
                    this: &Self,
                    accumulator: &mut Accumulator<P>,
                    other: &Accumulator<P>,
                ) {
                    Self::$core(this).fuse_accumulators::<P>(accumulator, other)
                }

                fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
                    Self::$core(this).output_mode()
                }

                fn to_output_parallel<Out: Numeric, Idx: Numeric>(
                    this: &Self,
                    accumulator: Accumulator<P>,
                    _shape_axis_reduce: usize,
                ) -> (Value<Out>, Value<Idx>) {
                    Self::$core(this).to_output_parallel::<P, Out, Idx>(accumulator)
                }

                fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
                    this: &Self,
                    accumulator: Accumulator<P>,
                    _shape_axis_reduce: usize,
                ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
                    Self::$core(this).to_output_perpendicular::<P, Out, Idx>(accumulator)
                }
            }
        };
    };
}

pub(crate) use forward_reduce_instruction;
