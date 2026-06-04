//! The coordinate space a tile lives in. An operation's space is the merge of
//! its operands' spaces; the axes the output drops are the ones it contracts.

use cubecl::zspace::SmallVec;

use crate::Partitioner;

use super::{Axis, ByAxis, Extent, MAX_AXES};

/// A coordinate space: every axis with its extent, ordered).
/// A tile lives in its own space (matmul's /// `lhs ∈ {M,K}`,
/// `rhs ∈ {K,N}`, `out ∈ {M,N}`); an operation ranges over their
/// [`merge`](Space::merge).
///
/// An extent is an [`Extent`]: usually a plain size, but a *composite* axis (a
/// squashed batch) keeps its factorization so broadcasting can act on a subset
/// of it.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Space {
    extents: ByAxis<Extent>,
    /// How this space decomposes — its stack of [`Partitioner`]s, coarse→fine. The
    /// space owns the tiling (it's the decomposition tree); a [`Tile`](crate::Tile)
    /// is a projection that binds data to it and carries no partitioner of its own.
    /// Each [`divide`](Space::divide) (and so each [`at`](crate::Tile::at))
    /// consumes the head level; when the stack is empty the space is a *leaf* (its
    /// extents are the final tile shape). Empty until the strategy attaches one or
    /// more levels at launch via [`with_partitioner`](Space::with_partitioner).
    partitioners: Vec<Partitioner>,
}

impl Space {
    /// Build from an ordered `(axis, extent)` list — the canonical axis order.
    /// Each axis is plain (single-level); use [`with_levels`](Space::with_levels)
    /// to declare a composite axis. No partitioner yet — add one with
    /// [`with_partitioner`](Space::with_partitioner).
    pub fn new(extents: &[(Axis, usize)]) -> Self {
        let entries = extents
            .iter()
            .map(|&(axis, n)| (axis, Extent::scalar(n)))
            .collect::<Vec<_>>();
        Space {
            extents: ByAxis::new(&entries),
            partitioners: Vec::new(),
        }
    }

    /// Build from an ordered `(axis, Extent)` list, allowing composite axes — a
    /// squashed batch declared as `Extent::new(&[4, 3])`, say.
    pub fn with_levels(extents: &[(Axis, Extent)]) -> Self {
        Space {
            extents: ByAxis::new(extents),
            partitioners: Vec::new(),
        }
    }

    /// Push a finer decomposition level: how this space splits into tiles. Called
    /// once for single-level tiling; chain it (coarse→fine) to declare multi-level
    /// tiling — each level can be a *totally different* [`Partitioner`], and the
    /// kernel descends them one [`divide`](Space::divide) at a time.
    pub fn with_partitioner(mut self, partitioner: Partitioner) -> Self {
        self.partitioners.push(partitioner);
        self
    }

    /// This space's head (coarsest) partitioner — the level a [`divide`](Space::divide)
    /// or [`Walk::over`](crate::Walk::over) cuts next (panics if the space is a leaf).
    pub fn partitioner(&self) -> &Partitioner {
        self.partitioners
            .first()
            .expect("Space: no partitioner attached (use `with_partitioner`)")
    }

    /// Whether this space is a leaf — no partitioner levels left to descend, so its
    /// extents *are* the final tile shape.
    pub fn is_leaf(&self) -> bool {
        self.partitioners.is_empty()
    }

    /// Flattened extent (size) of the space along an axis — the total the walk
    /// and partitioner see, regardless of whether the axis is composite.
    pub fn extent(&self, axis: Axis) -> usize {
        self.extents.get(axis).total()
    }

    /// The structured extent along an axis, including a composite axis's levels.
    pub fn levels(&self, axis: Axis) -> Extent {
        self.extents.get(axis)
    }

    /// The axis at position `i` in canonical order.
    pub fn axis_at(&self, i: usize) -> Axis {
        self.extents.axis_at(i)
    }

    /// The canonical-order position of `axis`.
    pub fn position(&self, axis: Axis) -> usize {
        self.extents.position(axis)
    }

    /// The space's dimension (number of axes).
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Whether `axis` is one of this space's axes.
    pub fn contains(&self, axis: Axis) -> bool {
        self.extents.contains(axis)
    }

    /// The smallest space containing every `part`: each axis carried by any part,
    /// in first-appearance order. A shared axis's extents are broadcast-merged
    /// per level ([`Extent::broadcast`]): `n ∪ n = n`, `1 ∪ n = n`, conflict
    /// otherwise. So an operand carrying batch `[4, 1]` and one carrying `[1, 3]`
    /// combine into the output's `[4, 3]`; an operand that omits an axis entirely
    /// broadcasts along all of it. E.g. `{M,K} ∪ {K,N} ∪ {M,N} = {M,N,K}`.
    pub fn merge(parts: &[&Space]) -> Space {
        let mut entries: SmallVec<[(Axis, Extent); MAX_AXES]> = SmallVec::new();
        for part in parts {
            let mut i = 0;
            while i < part.rank() {
                let axis = part.axis_at(i);
                let extent = part.levels(axis);
                match entries.iter_mut().find(|(a, _)| *a == axis) {
                    Some(slot) => slot.1 = slot.1.broadcast(extent),
                    None => entries.push((axis, extent)),
                }
                i += 1;
            }
        }
        // Operands of one operation share its partitioner stack, so the merge
        // carries the first part's. (A contracted axis like K is therefore tiled
        // once, at every level.)
        let partitioners = parts
            .iter()
            .map(|p| &p.partitioners)
            .find(|s| !s.is_empty())
            .cloned()
            .unwrap_or_default();
        Space {
            extents: ByAxis::new(&entries),
            partitioners,
        }
    }

    /// Project onto `axes` (in the given order): the sub-Space carrying only those
    /// axes, extents copied from self — composite axes keep their levels. Dropping
    /// the rest is the only axis-changer; the projection keeps self's partitioner,
    /// so a tile over `{M,K}` reads the same tiling its operation decided. Fired at
    /// setup to build each operand's tile geometry (`lhs` projects onto `{M,K}`, …).
    pub fn project(&self, axes: &[Axis]) -> Space {
        let mut sub = Space::with_levels(
            &axes
                .iter()
                .map(|&a| (a, self.levels(a)))
                .collect::<Vec<_>>(),
        );
        sub.partitioners = self.partitioners.clone();
        sub
    }

    /// Tiles along `axis`: `extent / sub-tile edge`. Both come from the space (it
    /// owns the partitioner), so the count of pieces a cut yields is a property of
    /// the space alone — no operand, no view. The per-axis counts are exactly what
    /// [`Walk::over`](crate::Walk::over) folds into a walk.
    pub fn count(&self, axis: Axis) -> usize {
        self.extent(axis) / self.partitioner().edge(axis)
    }

    /// The axes in this space but not in `output` — those an operation contracts.
    /// Matmul over `{M,N,K}` with output `{M,N}` contracts `{K}`.
    pub fn contracting(&self, output: &Space) -> SmallVec<[Axis; MAX_AXES]> {
        let mut contracted = SmallVec::new();
        let mut i = 0;
        while i < self.rank() {
            let axis = self.axis_at(i);
            if !output.contains(axis) {
                contracted.push(axis);
            }
            i += 1;
        }
        contracted
    }

    /// The axes of this space in canonical order.
    pub fn axes(&self) -> Axes<'_> {
        Axes { space: self, i: 0 }
    }

    /// The child space this one divides into — the tiling schema one level down
    /// (every axis shrunk to its head partitioner's sub-tile edge), with that head
    /// level **consumed** so the child carries the remaining (finer) stack.
    /// Position-free: the shape is congruent for every piece of the cut, so it's
    /// the comptime *shape* half of a subdivision; the *positions* are the [`Walk`]
    /// (see [`Walk::over`](crate::Walk::over)). Repeatedly dividing walks the
    /// decomposition tree down to the [`leaf`](Space::leaf). A composite (squashed)
    /// axis keeps its factorization through the cut (see [`Extent::sub_extent`]), so
    /// broadcasting stays level-aligned at deeper levels.
    pub fn divide(&self) -> Space {
        let head = self.partitioner();
        let mut child = Space::with_levels(
            &self
                .axes()
                .map(|axis| (axis, self.levels(axis).sub_extent(head.edge(axis))))
                .collect::<Vec<_>>(),
        );
        child.partitioners = self.partitioners[1..].to_vec();
        child
    }

    /// The leaf space this one bottoms out in — divide repeatedly until no
    /// partitioner level is left. Its extents are the final (finest) tile shape;
    /// used to size the staging buffers and to read the leaf's `mr`/`nr`/`kc`.
    pub fn leaf(&self) -> Space {
        let mut space = self.clone();
        while !space.is_leaf() {
            space = space.divide();
        }
        space
    }

    /// Element count of this space's tile — the product of every axis's extent.
    /// On a [`leaf`](Space::leaf) this sizes the shared-memory staging buffer.
    pub fn tile_size(&self) -> usize {
        self.axes().map(|axis| self.extent(axis)).product()
    }
}

/// Iterator over a [`Space`]'s axes in canonical order. See [`Space::axes`].
pub struct Axes<'a> {
    space: &'a Space,
    i: usize,
}

impl Iterator for Axes<'_> {
    type Item = Axis;

    fn next(&mut self) -> Option<Axis> {
        if self.i < self.space.rank() {
            let axis = self.space.axis_at(self.i);
            self.i += 1;
            Some(axis)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.space.rank() - self.i;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Axes<'_> {}

impl<'a> IntoIterator for &'a Space {
    type Item = Axis;
    type IntoIter = Axes<'a>;

    fn into_iter(self) -> Axes<'a> {
        self.axes()
    }
}
