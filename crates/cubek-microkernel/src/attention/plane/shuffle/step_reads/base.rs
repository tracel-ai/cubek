use cubecl::prelude::*;
use cubecl::tile::Region;
use cubecl::tile::layout::PageEntry;
use cubecl::{self as cubecl, cube};

/// Where one unit stands in a step: its team and its lane in the team, the keys attended, and
/// how the plane's teams share the step's keys, `keys_in_flight` each, `units_per_key` units on
/// each key.
#[derive(CubeType, Clone)]
pub struct StepWalk {
    pub team: usize,
    pub lane: usize,
    pub attended: u32,
    #[cube(comptime)]
    pub teams: usize,
    #[cube(comptime)]
    pub keys_in_flight: usize,
    #[cube(comptime)]
    pub units_per_key: usize,
}

/// The lines a step's teams load from one operand, keys or values: the unit's lines of each of
/// its team's keys of `step`, every load issued before any is used. A key past the attended ones
/// reads nothing, and neither does a line past `lines`.
#[cube]
pub trait StepReads<N: Size>: CubeType + Sized {
    /// The element its lines arrive in: the operand's own, or what a packed one decodes to.
    type Element: Float;

    fn load(
        &self,
        step: &Region,
        walk: &StepWalk,
        #[comptime] lines: usize,
    ) -> Array<Vector<Self::Element, N>>;

    /// These lines cut to `region`'s window, so the steps inside it are cut from there rather
    /// than from the operand's top.
    fn at(&self, region: &Region) -> Self;

    /// The page table entry `region`'s window is cut with, read ahead of the step that cuts it
    /// and handed to that step's lines ([`with_page_entry`](Self::with_page_entry)).
    /// [`PageEntry::unlisted`] where these lines are not paged.
    fn read_page_entry(&self, region: &Region) -> PageEntry;

    /// These lines taking their pages from `entry` where a cut names its sequence and column.
    /// `entry` must come from lines reading the same page table, as keys and values do.
    fn with_page_entry(&self, entry: &PageEntry) -> Self;
}
