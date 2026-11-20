use crate::matmul::cube::MatmulPlaneLayout;
use crate::matmul::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentRowsAlong};
use core::num::NonZeroUsize;
use cubecl::tile::Scratch;
use cubecl::tile::space::ComputeScope;
use cubek_std::microkernel::{StageLayout, StageRing};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedMicrokernelSettings {
    pub plane: PlaneFragmentMicrokernelSettings,
    /// How its planes are laid out over the box, `rows` along `m`, `columns` along `n`.
    pub planes: MatmulPlaneLayout,
    /// Planes sharing each tile, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
    /// How much of a plane's fragments the window it bounces them through holds at once, where
    /// they bounce: one fragment, three cube barriers each, or all of them, two barriers for the
    /// whole drain at as many times the bytes.
    pub scratch: Scratch,
    /// How it holds the stages its planes contract.
    pub ring: StageRing,
    /// How a stage its units copy lays out the rows its fragments load, the tile arranging its
    /// rows as the instruction reads them: swizzled tiles are padded under `cmma`, which reads a
    /// tile off a pointer and a stride (`PlaneFragmentMicrokernel::stage_storage`). A box the TMA
    /// engine lands lies as the instruction reads it whatever this says. Which reads faster
    /// depends on the device and on which way the operands run, so it is a setting a table races
    /// rather than a rule.
    pub stage_layout: StageLayout,
    /// Steps of `K` one stage holds against a packed input its units decode, each a step of the
    /// inputs' finest reads: a scale block where scales are bytes, a word of scales otherwise. The
    /// stage then reads each input as widely as it holds whole: four steps over byte scales are a
    /// word of them, read a word at a time; two MXFP4 blocks are 64 of `K`, read a scale at a time.
    /// Against float inputs a stage is its planes' step, which holds no scale block to count.
    pub stage_steps: NonZeroUsize,
}

impl CubeStagedMicrokernelSettings {
    /// A box of four by four fragments on each of two by four planes, one stage at a time.
    pub const DEFAULT: Self = Self {
        plane: PlaneFragmentMicrokernelSettings::DEFAULT,
        planes: MatmulPlaneLayout::new(2, 4),
        planes_along_k: NonZeroUsize::MIN,
        scratch: Scratch::OneTile,
        ring: StageRing::SINGLE,
        stage_layout: Self::SWIZZLED,
        stage_steps: NonZeroUsize::MIN,
    };

    /// The layout a stage takes where a table names none: tiles of the instruction, their rows'
    /// chunks swizzled, which `cmma` reads padded. Both keep the rows a fragment load reads at
    /// once on distinct banks, each the way its transport can read; the swizzle spends no shared
    /// memory. A block-scaled instruction's stage holds the stored words in the same tiles: a row
    /// of its tile is the instruction's `k` of words, read eight rows at a time as a 16-bit
    /// instruction's are.
    ///
    /// Measured against padding under the manual `mma`: an L4's shared loads go from conflicted to
    /// conflict-free at a level time; gfx1151's bank conflicts fall to zero, 8 to 20% faster on
    /// four of six boxes, the others within noise (a stage 64 deep along a col rhs's `K` up to
    /// about 10% slower).
    pub const SWIZZLED: StageLayout = StageLayout::SwizzledBlocks;

    /// One tile a cube, four planes splitting its `K`: what a problem with too few output tiles to
    /// fill the device wants, each tile getting a cube's planes rather than one plane. The tile is
    /// half as high as the default's plane's, for what the scratch and the accumulator leave of
    /// shared memory.
    pub const SPLIT_ALONG_K: Self = Self {
        plane: PlaneFragmentMicrokernelSettings::grid(2, 4, 1),
        planes: MatmulPlaneLayout::new(1, 1),
        planes_along_k: NonZeroUsize::new(4).unwrap(),
        ..Self::DEFAULT
    };

    /// A box against a packed input, counted in steps of the packed product
    /// ([`MatmulStep`](crate::matmul::MatmulStep)): `steps` of `M` and `N` on each
    /// plane, on `planes` (along `m` by along `n`), holding its stages in `ring`. A step's box is
    /// whole loads of the packed input's words and scales, so every load filling a stage is
    /// whole; the stage holds the step's scale blocks, decoded by the units that fill it, and each
    /// plane drains its own tile.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn in_steps(steps: (usize, usize), planes: (usize, usize), ring: StageRing) -> Self {
        Self {
            plane: PlaneFragmentMicrokernelSettings {
                along_m: NonZeroUsize::new(steps.0).unwrap(),
                along_n: NonZeroUsize::new(steps.1).unwrap(),
                ..PlaneFragmentMicrokernelSettings::DEFAULT
            },
            planes: MatmulPlaneLayout::new(planes.0, planes.1),
            planes_along_k: NonZeroUsize::MIN,
            scratch: Scratch::None,
            ring,
            stage_layout: Self::SWIZZLED,
            stage_steps: NonZeroUsize::MIN,
        }
    }

    /// A box against a packed input its stages hold as it is stored, its planes decoding it as
    /// each fragment loads: `fragments` of the decoding instruction along `m`, `n` and `k` on each
    /// plane, each fragment as deep as the stage
    /// ([`windowed`](PlaneFragmentMicrokernelSettings::windowed)), on `planes` (along `m` by along
    /// `n`), holding its stages in `ring`. Each fragment of the rhs is decoded once a step and
    /// contracted against every fragment of the lhs along `m`.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn stored(
        fragments: (usize, usize, usize),
        planes: (usize, usize),
        ring: StageRing,
    ) -> Self {
        Self {
            plane: PlaneFragmentMicrokernelSettings {
                along_m: NonZeroUsize::new(fragments.0).unwrap(),
                ..PlaneFragmentMicrokernelSettings::windowed(fragments.1, fragments.2)
            },
            planes: MatmulPlaneLayout::new(planes.0, planes.1),
            planes_along_k: NonZeroUsize::MIN,
            scratch: Scratch::OneTile,
            ring,
            stage_layout: Self::SWIZZLED,
            stage_steps: NonZeroUsize::MIN,
        }
    }

    /// Eight by one steps on each of one by two planes, one stage at a time: on the M2, eight by
    /// two instructions a plane in tile order, the float staged routine's fastest box on the
    /// Qwen3-4B projections at a prompt's rows, whose planes contract the same stages these hold.
    pub const PACKED: Self = Self::in_steps((8, 1), (1, 2), StageRing::SINGLE).stage_steps(4);

    /// These settings with their planes' instruction laying its rows along `rows_along`.
    pub const fn rows_along(self, rows_along: PlaneFragmentRowsAlong) -> Self {
        Self {
            plane: self.plane.rows_along(rows_along),
            ..self
        }
    }

    /// These settings with `planes` planes sharing each box along `K`.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn split_along_k(self, planes: usize) -> Self {
        Self {
            planes_along_k: NonZeroUsize::new(planes).unwrap(),
            ..self
        }
    }

    /// `plane` on each of `planes`, the default's other settings.
    pub const fn boxed(plane: PlaneFragmentMicrokernelSettings, planes: MatmulPlaneLayout) -> Self {
        Self {
            plane,
            planes,
            ..Self::DEFAULT
        }
    }

    /// These settings with each stage holding `steps` steps of `K` against a packed input.
    ///
    /// # Panics
    ///
    /// On a count of zero.
    pub const fn stage_steps(self, steps: usize) -> Self {
        Self {
            stage_steps: NonZeroUsize::new(steps).unwrap(),
            ..self
        }
    }

    /// These settings holding their stages in `ring`.
    pub const fn ringed(self, ring: StageRing) -> Self {
        Self { ring, ..self }
    }

    /// These settings with each plane tile held by `holder`.
    pub const fn held_by(self, holder: ComputeScope) -> Self {
        Self {
            plane: self.plane.held_by(holder),
            ..self
        }
    }

    /// Whether these settings hold the same box of the output as `other`: the same plane grid on
    /// the same layout of planes, whatever their stages.
    pub fn same_box(&self, other: &Self) -> bool {
        (self.plane, self.planes) == (other.plane, other.planes)
    }
}

/// Nothing for one plane along `K`, a one-fragment scratch, the swizzled stage layout or a stage
/// of one step, so the settings that hold them keep their
/// names.
impl core::fmt::Display for CubeStagedMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} {}", self.plane, self.planes)?;
        if self.planes_along_k.get() > 1 {
            write!(f, " planes_along_k={}", self.planes_along_k)?;
        }
        if self.scratch != Scratch::OneTile {
            write!(f, " scratch={}", self.scratch)?;
        }
        write!(f, "{}", self.ring)?;
        if self.stage_layout != Self::SWIZZLED {
            write!(f, "{}", self.stage_layout)?;
        }
        if self.stage_steps.get() > 1 {
            write!(f, " stage_steps={}", self.stage_steps)?;
        }
        Ok(())
    }
}
