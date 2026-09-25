//! A window addressed from its origin and strides, for a reader that builds its own offsets:
//! the [`WindowAddress`] and why a window may have none, the address itself and the row stride
//! a fragment steps by, then the buffer and the views a reader of the address takes it through.

use cubecl::{
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut, layout::CoordsDyn,
    },
    unexpanded,
};

use crate::*;

/// Where a window's lines sit in its buffer: the line its origin is at and, per coordinate, how
/// many lines one step along it moves. The line at `pos` is `origin + Σ pos[c] · strides[c]`.
///
/// A layout walk derives every read's offset from its coordinates again, and how much of that the
/// backend folds or moves out of a loop depends on what it knows at compile time: a dynamic extent
/// can leave a multiply on every read. A reader holding this builds its offsets itself, so it can
/// take the constant parts once and advance a running index along its loop.
///
/// The coordinates are the window's, which are the tile's own axes on a direct store and the
/// buffer's positions on a split one (a [`Disjoint`](Composition::Disjoint) partition of an axis
/// is resolved above the layout). Only a window whose every line sits at that offset has one
/// ([`MemData::window_address`]).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct WindowAddress {
    /// The line the window's origin sits at.
    pub(crate) origin: u32,
    /// Per window coordinate, in lines.
    pub(crate) strides: Coords<u32>,
}

/// Why a window has no [`WindowAddress`], or why its lines are not its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unaddressable {
    /// Read or written through a call, which has no address.
    Call,
    /// A gather: its cells as they lie are not its matrix (a stage holds the compacted physical
    /// window, and the leaf gathers on read), and sibling windows overlap.
    Gathered,
    /// Some of its reads are masked at an overhang, which only the layout walk tests.
    Masked,
    /// It spans several storage tiles, the tile of this level (`None`: of no level), so its offsets
    /// jump at each boundary.
    SpansStorageTiles(Option<usize>),
    /// Its lines hold packed or quantized words, which only a decoding read serves as values.
    Packed,
}

impl Unaddressable {
    fn refuse(self, site: &str) -> ! {
        match self {
            Unaddressable::Call => panic!(
                "{site}: this tile's backing is a call, which has no address; it is reached \
                 through its layout"
            ),
            Unaddressable::Gathered => panic!(
                "{site}: a gathered window's cells as they lie are not its matrix; read it \
                 through its layout"
            ),
            Unaddressable::Masked => {
                panic!("{site}: this window masks an overhang, which only its layout walk tests")
            }
            // Offsets from a base and strides would walk straight through a storage tile boundary
            // and return another tile's cells, silently.
            Unaddressable::SpansStorageTiles(Some(level)) => panic!(
                "{site}: this window sits above its storage tile (the tile of level {level}), \
                 spanning several, so it is not one contiguous region; descend through that level \
                 first, or read the operand through its layout"
            ),
            Unaddressable::SpansStorageTiles(None) => panic!(
                "{site}: this operand's storage tile is the tile of no level of the kernel's nest, \
                 so no window is known to lie inside one; read it through its layout, or stage it"
            ),
            Unaddressable::Packed => panic!(
                "{site}: a packed store's lines are words, not plain values; a fragment load \
                 reads it through Tile::matrix_transparent, anything else through its packed views"
            ),
        }
    }
}

impl<T: Numeric> MemDataExpand<T> {
    /// The one test behind every [`WindowAddress`]: `require_plain` asks also that the lines be
    /// the plain values they serve ([`Packing::Plain`]).
    fn unaddressable(&self, require_plain: bool) -> Option<Unaddressable> {
        if !matches!(self.store.backing, BackingExpand::Buffer(_)) {
            return Some(Unaddressable::Call);
        }
        // A gathered operand is never storage-tiled (`Projection::validate`), so this never
        // refuses a window inside one storage tile, the one a fragment loads as it lies.
        if self.projection.composition() == Composition::Overlapping {
            return Some(Unaddressable::Gathered);
        }
        if self.access.overhang.masks() {
            return Some(Unaddressable::Masked);
        }
        if let Storage::Tiled(level) = self.access.storage {
            return Some(Unaddressable::SpansStorageTiles(level));
        }
        if require_plain && self.store.packing != Packing::Plain {
            return Some(Unaddressable::Packed);
        }
        None
    }

    // Written as expand methods by hand: the test matches the backing's expand variant and
    // panics at expansion with the caller's site, neither of which a `#[cube]` body can do, since
    // the macro also compiles that body unexpanded, against `MemData` rather than its expand.
    pub(crate) fn __expand_addressed_by_axes_method(&self, _scope: &Scope) -> bool {
        self.unaddressable(true).is_none() && self.projection.is_direct()
    }

    pub(super) fn __expand_refuse_unaddressable_method(&self, _scope: &Scope, site: &str) {
        if let Some(reason) = self.unaddressable(false) {
            reason.refuse(site)
        }
    }

    pub(super) fn __expand_refuse_unaddressable_plain_method(&self, _scope: &Scope, site: &str) {
        if let Some(reason) = self.unaddressable(true) {
            reason.refuse(site)
        }
    }
}

impl<T: Numeric> MemData<T> {
    /// Whether a reader may index this window's plain values by the tile's own axes: it has a
    /// [`WindowAddress`], its lines hold unpacked values, and its store is direct, so the
    /// address's coordinates are those axes.
    pub(crate) fn addressed_by_axes(&self) -> bool {
        unexpanded!()
    }

    /// Refuse at expansion, naming `site`, a window that has no [`WindowAddress`].
    pub(super) fn refuse_unaddressable(&self, _site: &str) {
        unexpanded!()
    }

    /// [`refuse_unaddressable`](MemData::refuse_unaddressable), also refusing a window whose
    /// lines are packed or quantized words rather than plain values.
    pub(super) fn refuse_unaddressable_plain(&self, _site: &str) {
        unexpanded!()
    }
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// This window's [`WindowAddress`], refused at expansion where it has none.
    ///
    /// A storage-tiled coordinate steps by its innermost fragment's stride, which holds inside one
    /// storage tile, the only place such a window is addressed.
    pub(crate) fn window_address(&self) -> WindowAddress {
        self.refuse_unaddressable(comptime!("MemData::window_address"));
        let positional = comptime!(self.layout.projection.clone());
        let mut strides = Coords::<u32>::new();
        #[unroll]
        for c in 0..comptime!(positional.coordinate_rank()) {
            let axis = comptime!(positional.logical_axes()[c]);
            let inner = comptime!(*positional.carriers(axis).last().unwrap());
            strides.push(self.layout.physical_strides.at(inner));
        }
        WindowAddress {
            origin: self.window_start,
            strides,
        }
    }

    /// Scalars between two rows of the window as a fragment reads it, `space` being the window's:
    /// its rows are the last axis of extent past one ([`MatrixAxes::edges`]), so a split
    /// contraction's block digit or a column tile's index does not stand between the fragment and
    /// its rows. The line stride widened back to scalars.
    ///
    /// A split store's coordinates are the buffer's positions, not the tile's axes, so no axis
    /// names its row; it keeps its own, the position above the one its trailing axes fold into.
    pub(crate) fn row_stride(&self, #[comptime] space: &Space) -> u32 {
        let address = self.window_address();
        let rank = address.strides.len();
        let row = comptime!(match self.projection.is_direct() {
            true => MatrixAxes::edges(space).row_split,
            false => rank - 2,
        });
        address
            .strides
            .at(row)
            .fmul(comptime!(self.store.vector_size as u32).runtime())
    }

    /// The whole buffer as `W`-wide lines of plain values, for a reader of
    /// [`window_address`](MemData::window_address). `W` is the store's own width, the one the
    /// address counts lines in.
    pub(crate) fn addressed_lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.refuse_unaddressable_lines::<W>();
        self.lines::<W>()
    }

    /// The mutable twin of [`addressed_lines`](MemData::addressed_lines).
    pub(crate) fn addressed_lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.refuse_unaddressable_lines::<W>();
        self.lines_mut::<W>()
    }

    fn refuse_unaddressable_lines<W: Size>(&self) {
        self.refuse_unaddressable_plain(comptime!("MemData::addressed_lines"));
        let width = W::value();
        comptime!(assert_eq!(
            width, self.store.vector_size,
            "MemData::addressed_lines: an address counts lines at the store's own width"
        ));
    }

    /// The buffer from this window's origin on, at the plain element it holds: what a fragment
    /// load or store addresses, stepping rows by [`row_stride`](MemData::row_stride). The origin
    /// counts lines, the buffer scalars.
    pub(crate) fn window_slice(&self) -> &[T] {
        self.refuse_unaddressable_plain(comptime!("MemData::window_slice"));
        let origin = self.window_origin_scalar();
        self.store.buffer().slice(origin, self.store.buffer().len())
    }

    /// The mutable twin of [`window_slice`](MemData::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        self.refuse_unaddressable_plain(comptime!("MemData::window_slice_mut"));
        let origin = self.window_origin_scalar();
        let end = self.store.buffer().len();
        self.store.buffer_mut().slice_mut(origin, end)
    }

    fn window_origin_scalar(&self) -> usize {
        (self.window_start * comptime!(self.store.vector_size as u32).runtime()) as usize
    }

    /// The window as one dense run of lines: index `i` addresses line `origin + i`, one add and no
    /// layout walk. Every line of the window must follow the one before it: a split store's or a
    /// storage-tiled store's never do, and are refused; a direct store's strides only say at run
    /// time, so that part is the caller's guarantee.
    pub(crate) fn dense_lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.refuse_undense();
        let origin = self.window_start as usize;
        let all = self.addressed_lines::<W>();
        all.slice(origin, all.len())
    }

    /// The mutable twin of [`dense_lines`](MemData::dense_lines).
    pub(crate) fn dense_lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.refuse_undense();
        let origin = self.window_start as usize;
        let all = self.addressed_lines_mut::<W>();
        let end = all.len();
        all.slice_mut(origin, end)
    }

    fn refuse_undense(&self) {
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::dense_lines: a split window's lines sit at the buffer's positions, not in \
             one run"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::dense_lines: a storage-tiled window's lines jump at each fragment"
        ));
    }

    /// Where a view over the window's own coordinates starts in the buffer's lines, and the layout
    /// it reads through: the window's extent, stepped by its address.
    fn addressed_run(&self) -> (usize, GmemLayout) {
        let address = self.window_address();
        let start = address.origin as usize;
        let layout = GmemLayout {
            physical_shape: self.window.extent.clone(),
            physical_strides: address.strides,
            projection: comptime!(Projection::direct(self.layout.projection.logical_axes())),
        };
        (start, layout)
    }

    /// The window as a view over its own coordinates, addressed from its origin.
    pub(super) fn addressed_view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        let (start, layout) = self.addressed_run();
        let all = self.lines::<W>();
        all.slice(start, all.len()).view(layout)
    }

    /// [`addressed_view`](MemData::addressed_view) at the storage element `I` the buffer truly
    /// holds, grouped `WP` wide.
    pub(super) fn addressed_view_storage<I: Numeric, WP: Size>(
        &self,
    ) -> View<'_, Vector<I, WP>, CoordsDyn> {
        let (start, layout) = self.addressed_run();
        let all = self.lines_storage::<I, WP>();
        all.slice(start, all.len()).view(layout)
    }

    /// The mutable twin of [`addressed_view`](MemData::addressed_view).
    pub(super) fn addressed_view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        let (start, layout) = self.addressed_run();
        let all = self.lines_mut::<W>();
        let len = all.len();
        all.slice_mut(start, len).view_mut(layout)
    }
}
