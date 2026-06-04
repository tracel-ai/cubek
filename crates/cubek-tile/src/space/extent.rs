//! The size of a space along one axis — a single number for a plain axis, or a
//! factorization for a *composite* one.

use cubecl::prelude::*;

/// Maximum number of levels a composite extent carries. Bump if a problem
/// squashes more dimensions into one axis than this.
pub const MAX_LEVELS: usize = 4;

/// The size of a space along one axis: a plain [`Scalar`](Extent::Scalar), or a
/// [`Composite`](Extent::Composite) factorization for a *squashed* axis (a batch
/// `[4, 3]`, say) that broadcasting can act on a subset of. The accessors
/// ([`len`](Extent::len) / [`level`](Extent::level) / [`weight`](Extent::weight))
/// treat a scalar as a single level, so callers iterate both uniformly.
///
/// `Copy` (and so usable in a [`ByAxis`](super::ByAxis)); a composite holds its
/// levels inline, the tail past `len` zeroed so derived `Eq`/`Hash` stay correct.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Extent {
    /// A plain axis — a single extent.
    Scalar(usize),
    /// A squashed axis carrying its ordered factorization (`len ≥ 2`).
    Composite {
        levels: [usize; MAX_LEVELS],
        len: u8,
    },
}

impl Extent {
    /// A plain (single-level) extent.
    pub fn scalar(n: usize) -> Self {
        Extent::Scalar(n)
    }

    /// An extent from ordered levels: a single level is a
    /// [`Scalar`](Extent::Scalar), more become a [`Composite`](Extent::Composite)
    /// (`[4, 3]` for a `4×3` batch).
    pub fn new(levels: &[usize]) -> Self {
        assert!(
            !levels.is_empty(),
            "Extent::new: at least one level required"
        );
        assert!(
            levels.len() <= MAX_LEVELS,
            "Extent::new: more than MAX_LEVELS levels"
        );
        if levels.len() == 1 {
            return Extent::Scalar(levels[0]);
        }
        let mut arr = [0; MAX_LEVELS];
        arr[..levels.len()].copy_from_slice(levels);
        Extent::Composite {
            levels: arr,
            len: levels.len() as u8,
        }
    }

    /// The flat extent the walk sees: the product of all levels.
    pub fn total(&self) -> usize {
        match self {
            Extent::Scalar(n) => *n,
            Extent::Composite { levels, len } => levels[..*len as usize].iter().product(),
        }
    }

    /// Number of levels (`1` for a scalar).
    pub fn len(&self) -> usize {
        match self {
            Extent::Scalar(_) => 1,
            Extent::Composite { len, .. } => *len as usize,
        }
    }

    /// Always `false` — an [`Extent`] always carries at least one level. Present
    /// to satisfy clippy alongside [`len`](Extent::len).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether the axis squashes more than one dimension.
    pub fn is_composite(&self) -> bool {
        matches!(self, Extent::Composite { .. })
    }

    /// The extent of level `i`.
    pub fn level(&self, i: usize) -> usize {
        match self {
            Extent::Scalar(n) => {
                assert!(i == 0, "Extent::level: index out of range");
                *n
            }
            Extent::Composite { levels, len } => {
                assert!(i < *len as usize, "Extent::level: index out of range");
                levels[i]
            }
        }
    }

    /// The row-major weight of level `i`: the product of every later level (`1`
    /// for a scalar). A flat index unravels as `coord_i = (flat / weight(i)) % level(i)`.
    pub fn weight(&self, i: usize) -> usize {
        match self {
            Extent::Scalar(_) => 1,
            Extent::Composite { levels, len } => levels[i + 1..*len as usize].iter().product(),
        }
    }

    /// The factorization of a row-major sub-block of `edge` elements taken from the
    /// front of this extent — the child extent a tiling level cuts this axis to.
    /// Levels are consumed innermost-first and the level count is **preserved** (so a
    /// composite axis stays broadcast-aligned across a multi-level descent):
    /// `[4, 3].sub_extent(3) == [1, 3]`, `.sub_extent(6) == [2, 3]`, `.sub_extent(12)
    /// == [4, 3]` (a pass-through), `.sub_extent(1) == [1, 1]`. A scalar yields a
    /// scalar (`n.sub_extent(e) == e`), so plain axes are unchanged. `edge` must
    /// factor cleanly into the trailing levels.
    pub fn sub_extent(&self, edge: usize) -> Extent {
        let n = self.len();
        let mut levels = [1usize; MAX_LEVELS];
        let mut rem = edge;
        let mut i = n;
        while i > 0 {
            i -= 1;
            let level = self.level(i);
            if rem >= level {
                assert!(
                    rem % level == 0,
                    "Extent::sub_extent: edge {edge} does not align to the levels of {self:?}"
                );
                levels[i] = level;
                rem /= level;
            } else {
                assert!(
                    level % rem == 0,
                    "Extent::sub_extent: edge {edge} does not align to the levels of {self:?}"
                );
                levels[i] = rem;
                rem = 1;
            }
        }
        assert!(
            rem == 1,
            "Extent::sub_extent: edge {edge} exceeds extent {}",
            self.total()
        );
        Extent::new(&levels[..n])
    }

    /// Merge two extents for the same axis under broadcasting: level by level,
    /// `n ∪ n = n` and `1 ∪ n = n`; a genuine `n ≠ m` mismatch (neither `1`)
    /// panics, as does a differing level count.
    pub fn broadcast(self, other: Extent) -> Extent {
        assert!(
            self.len() == other.len(),
            "Extent::broadcast: axis appears with differing level counts"
        );
        if let (Extent::Scalar(a), Extent::Scalar(b)) = (self, other) {
            return Extent::Scalar(merge_level(a, b));
        }
        let mut levels = [0; MAX_LEVELS];
        let mut i = 0;
        while i < self.len() {
            levels[i] = merge_level(self.level(i), other.level(i));
            i += 1;
        }
        Extent::Composite {
            levels,
            len: self.len() as u8,
        }
    }
}

/// One level's broadcast rule: equal sizes agree, a `1` yields to the other,
/// anything else conflicts.
fn merge_level(a: usize, b: usize) -> usize {
    match (a, b) {
        (1, b) => b,
        (a, 1) => a,
        (a, b) if a == b => a,
        _ => panic!("Extent::broadcast: axis appears with conflicting extents"),
    }
}

/// Map a flat coordinate in `from`'s flattened space to the corresponding flat
/// coordinate in `into`'s, broadcasting on every level `into` carries as size 1.
#[cube]
pub fn project_flat(#[comptime] into: Extent, #[comptime] from: Extent, flat: usize) -> usize {
    comptime!(assert!(
        into.len() == from.len(),
        "Extent::project_flat: level-count mismatch between operand and operation extent"
    ));

    let mut index = 0;
    #[unroll]
    for r in 0..comptime!(from.len()) {
        let level = comptime!(from.len() - 1 - r);
        let coord = (flat / comptime!(from.weight(level))) % comptime!(from.level(level));
        let placed = coord
            * comptime!(if into.level(level) == 1 {
                0usize
            } else {
                1usize
            });
        index += placed * comptime!(into.weight(level));
    }
    index
}
