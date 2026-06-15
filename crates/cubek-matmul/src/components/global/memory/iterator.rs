use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};

#[derive(Clone, CubeType)]
#[expand(derive(Clone))]
/// An iterator over global memory, advancing along k.
pub struct GlobalIterator<'a, EI: CubePrimitive> {
    global_view: View<'a, EI, Coords2d>,
    offset: RuntimeCell<u32>,
    /// The amount to advance by on each iteration
    step: u32,
    view_size: Coords2d,
    #[cube(comptime)]
    view_direction: ViewDirection,
    #[cube(comptime)]
    checked: bool,
}

unsafe impl<EG: CubePrimitive> Sync for GlobalIterator<'_, EG> {}
unsafe impl<EG: CubePrimitive> Send for GlobalIterator<'_, EG> {}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum ViewDirection {
    Row,
    Col,
    /// Cannot advance if direction is none
    #[default]
    None,
}

#[cube]
impl<'a, EG: CubePrimitive> GlobalIterator<'a, EG> {
    /// Instantiate a read iterator over the given global view, which should be sliced to the size
    /// of one `m`/`n` stage and the full range of `k` handled by this matmul instance.
    ///
    /// `step` is the amount advanced in `view_direction` each iteration.
    /// `checked` determines whether the slices should be created as checked or unchecked.
    pub fn new(
        global_view: View<'a, EG, Coords2d>,
        step: u32,
        #[comptime] view_direction: ViewDirection,
        #[comptime] checked: bool,
    ) -> Self {
        let (size_row, size_col) = global_view.shape();
        let view_size = match view_direction {
            ViewDirection::Row => (step, size_col),
            ViewDirection::Col => (size_row, step),
            ViewDirection::None => (size_row, size_col),
        };

        GlobalIterator::<'a, EG> {
            global_view,
            offset: RuntimeCell::new(0),
            step,
            view_size,
            view_direction,
            checked,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn advance(&self) {
        self.offset.store(self.offset.read() + self.step);
    }

    /// Returns the current view slice of the iterator
    pub fn view(&self) -> View<'_, EG, Coords2d> {
        let offset = match self.view_direction.comptime() {
            ViewDirection::Row => (self.offset.read(), 0u32),
            ViewDirection::Col => (0u32, self.offset.read()),
            ViewDirection::None => (0u32, 0u32).runtime(),
        };
        if self.checked.comptime() {
            self.global_view.slice(offset, self.view_size)
        } else {
            self.global_view.slice_unchecked(offset, self.view_size)
        }
    }

    /// Returns the vector size of the global view
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        self.global_view.vector_size()
    }

    pub fn offset(&self) -> u32 {
        self.offset.read()
    }
}
