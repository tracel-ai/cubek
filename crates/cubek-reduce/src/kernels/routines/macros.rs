//! What the reduce routines' launch functions, launches and choices of microkernel share.

/// The launch function `$kernel`, whose owners reduce their rows by `$reduce`, what it needs of every
/// owner computed by `$before`; `$operands` are the operands it takes beside the input and the
/// outputs.
macro_rules! reduce_kernel {
    (
        $(#[$doc:meta])*
        $kernel:ident[$($operands:tt)*]: $microkernel_type:ty,
        |$microkernel:ident, $partitioning:ident, $owner:ident, $x:ident, $o:ident, $i:ident|
        { $($before:tt)* }
        { $($reduce:tt)* }
    ) => {
        $(#[$doc])*
        /// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller
        ///   binds in its place (`I`).
        /// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
        /// - `indices`: one tensor of `out`'s shape at its own strides, where the launch writes the
        ///   indices.
        #[cube(launch_unchecked)]
        pub(crate) fn $kernel<
            In: Numeric,
            Acc: Numeric,
            Out: Numeric,
            Idx: Numeric,
            N: Size,
            K: Size,
            F: ReduceCarrierFamily,
            I: TileOperand,
        >(
            input: &I::Arg,
            out: &mut TileTensor<'_>,
            indices: ComptimeOption<&mut TileTensor<'static>>,
            $($operands)*
            $partitioning: Partitioning,
            #[comptime] $microkernel: $microkernel_type,
            #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
        ) {
            let $x = I::tile::<In>(input, &$partitioning);
            let $o = out.tile_mut(&$partitioning);
            let $i: ComptimeOption<Tile> =
                indices.map(|indices| indices.tile_mut(&$partitioning));
            $($before)*
            for $owner in &$partitioning {
                $($reduce)*
            }
        }
    };
}

/// `$kernel`, a [`reduce_kernel`], launched on `$grid` as `$blueprint` states it for `$problem`, over
/// the operands `$binding` holds and the `$operand`s it takes beside them.
macro_rules! launch_reduce_kernel {
    (
        $kernel:ident::<$family:ident, $input:ident>,
        $problem:expr, $client:expr, $binding:expr, $blueprint:expr, $grid:expr
        $(, $operand:expr)* $(,)?
    ) => {
        // SAFETY: every tensor is bound through the launcher, which masks the reads and writes of a
        // tile the problem's edge cuts short, and the kernel indexes no buffer but through its
        // tiles. The shared memory it holds is the slots its microkernel declares, which the grid
        // counted. What a kernel does with the operands it takes beside these is stated where it is
        // launched.
        unsafe {
            $kernel::launch_unchecked::<$family, $input>(
                $client,
                $grid.launch.cube_count,
                $grid.launch.cube_dim,
                $blueprint.lane_count,
                $problem.operation.written_per_row(),
                $binding.input,
                $binding.out,
                $binding.indices.map(Bound::arg).into(),
                $($operand,)*
                $binding.launcher.partitioning_arg(),
                $blueprint.microkernel,
                $problem.compiled_elements(),
            )
        }
    };
}

/// The best `$microkernel` for `$problem` on `$hw` at `$settings`, and the lanes it binds the input
/// at.
macro_rules! best_microkernel {
    ($microkernel:ty, $hw:expr, $problem:expr, $settings:expr) => {{
        let microkernel = <$microkernel>::best($hw, &$problem.work_item(), &$settings.microkernel)?;
        let lane_count = Self::lane_count($hw, $problem, &microkernel, microkernel.widest_line())?;
        (microkernel, lane_count)
    }};
}
