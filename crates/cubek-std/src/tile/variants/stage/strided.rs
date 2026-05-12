use cubecl::{prelude::*, std::Swizzle};

use crate::{
    stage::StageMemoryConfig,
    tile::variants::stage::{TilingLayout, TilingLayoutEnum, memory::StridedStageMemory},
};

/// Type-erased view of a [`StridedStageMemory`] buffer for use as a tile kind.
///
/// Mirrors [`SharedTile`](crate::tile::variants::SharedTile)'s pattern: the
/// vector-size (`NS`) and tiling-layout (`T`) generics that parameterize the
/// underlying allocation are dropped at the type level. The vector size lives
/// in `config.vector_size`, the tiling layout is encoded as a comptime
/// [`TilingLayoutEnum`], and the smem slice is held as a scalar `Slice<E, IO>`
/// (re-typed via `downcast_unchecked` at construction).
///
/// Lifecycle: the underlying `StridedStageMemory` continues to own the
/// allocation; `StridedStage` is a non-owning view installed as a
/// [`TileKind::Stage`](crate::tile::TileKind) payload.
#[derive(CubeType, Clone, Copy)]
pub struct StridedStage<E: Numeric, IO: SliceVisibility = ReadOnly> {
    /// Scalar-typed slice covering the active buffer of the underlying smem.
    /// Re-typed back to `Slice<Vector<E, NS>, IO>` at lookup time via
    /// `with_vector_size::<NS>()` using `config.vector_size`.
    pub smem: Slice<E, IO>,
    pub swizzle: Swizzle,
    #[cube(comptime)]
    pub config: StageMemoryConfig,
    #[cube(comptime)]
    pub tiling_layout: TilingLayoutEnum,
}

#[cube]
impl<E: Numeric> StridedStage<E, ReadOnly> {
    /// Wraps a `StridedStageMemory`'s active buffer as a type-erased view.
    /// `NS` and `T` are consumed here and become runtime/comptime metadata.
    pub fn wrap<NS: Size, T: TilingLayout>(
        stage: &StridedStageMemory<E, NS, T>,
    ) -> StridedStage<E, ReadOnly> {
        let typed = stage.as_slice::<NS>();
        let erased: Slice<E, ReadOnly> = unsafe { typed.downcast_unchecked::<E>() };
        StridedStage::<E, ReadOnly> {
            smem: erased,
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}

#[cube]
impl<E: Numeric> StridedStage<E, ReadWrite> {
    /// Mutable variant of [`StridedStage::wrap`].
    pub fn wrap_mut<NS: Size, T: TilingLayout>(
        stage: &mut StridedStageMemory<E, NS, T>,
    ) -> StridedStage<E, ReadWrite> {
        let typed = stage.as_slice_mut::<NS>();
        let erased: SliceMut<E> = unsafe { typed.downcast_unchecked::<E>() };
        StridedStage::<E, ReadWrite> {
            smem: erased,
            swizzle: stage.swizzle,
            config: comptime!(stage.config),
            tiling_layout: comptime!(T::to_enum()),
        }
    }
}
