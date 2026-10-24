use crate::matmul::MatmulPartitioning;
use crate::matmul::dense::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedMicrokernelSettings, CubeStagedRows,
    CubeStagedStore, CubeStagedWalk, CubeStagedWgmmaRefusal,
};
use crate::matmul::dense::plane::{
    PlaneFragmentDrain, PlaneFragmentInstruction, PlaneFragmentMicrokernel,
    PlaneFragmentMicrokernelSettings, PlaneFragmentTransport,
};
use crate::matmul::staged_walk::MatmulStage;
use crate::matmul::staged_walk::{MatmulCubeWrite, MatmulPlaneLayout, MatmulStagingCount};
use crate::matmul::stored::BlockScales;
use crate::matmul::{
    MatmulAxes, MatmulBuffers, MatmulDestination, MatmulMicrokernel, MatmulOperand, MatmulOperands,
    MatmulOrder, MatmulRead, MatmulRowsKind, MatmulStep, MatmulWorkItem,
};
use crate::staging::StageRing;
use core::num::NonZeroUsize;
use cubecl::tile::launch::Delivery;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Scratch, Space, StageStorage, launch::Bound};
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, KB, M, N};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A box of the output one cube owns: it stages the box's rows of the lhs and columns of the rhs
/// a step of `K` at a time, and its planes each run a [`PlaneFragmentMicrokernel`] out of every
/// stage.
///
/// Its planes lie over the box along `m` and `n`, each owning a tile of it: independent work,
/// sharing nothing but the stage the cube fills for all of them. They may also lie along `K`,
/// each contracting its own slice of every step of the same tile, their partials merged in
/// shared memory, atomically: the cube's planes splitting `K` ([`CubeStagedDrain`]).
///
/// Its settings, with the plane's settings replaced by the plane microkernel the hardware
/// settled. The scratch, the ring and the fill are the settings' requests until it is resolved
/// against the box a routine places it in ([`resolve`](Self::resolve)), and the write is how the
/// routine's cubes reach the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedMicrokernel {
    /// What each of its planes runs.
    pub plane: PlaneFragmentMicrokernel,
    /// Its planes along `m` and `n`, each owning its own tile of the box.
    pub planes: MatmulPlaneLayout,
    /// Planes sharing each tile, each on its own slice of every step of `K`.
    pub planes_along_k: NonZeroUsize,
    /// The window each plane's fragments bounce through where they cannot land on their own.
    pub scratch: Scratch,
    /// How it holds the stages its planes contract.
    pub ring: StageRing,
    /// Who fills them.
    pub fill: CubeStagedFill,
    /// How a stage lays out the rows its fragments load.
    pub stage_rows: CubeStagedRows,
    /// Who stores its box of the output.
    pub store: CubeStagedStore,
    /// What one stage spans along the contraction.
    pub depth: CubeStagedDepth,
    /// How its planes read each operand out of the stages.
    pub reads: MatmulOperands<MatmulRead>,
}

/// What one stage of a cube spans along the contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedDepth {
    /// Its planes' steps of `K`.
    Steps,
    /// This many scale blocks of a packed input along `KB`, its planes' steps walking inside each
    /// one: the stage of a product over a block-split space.
    ScaleBlocks(NonZeroUsize),
}

/// How a cube microkernel's planes reach its output once the walk ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeStagedDrain {
    /// One plane along `K`: each plane drains its own tile, through a scratch window where its
    /// fragment cannot land its cells on its own.
    EachPlane,
    /// Several along `K`: every plane adds its partial of the tile into one accumulator in shared
    /// memory, atomically, and the cube copies it out once it has synchronized.
    Merged,
}

impl CubeStagedMicrokernel {
    /// What a packed item whose scales do not group `K` is declined with.
    pub const UNGROUPED_K: Refusal = Refusal::Declined {
        why: "the packed staged gemm stages K a scale block at a time, and this packed input's scales do not group K",
    };

    /// Boxes against a packed input several steps of `M` a plane, `(steps, planes)` as
    /// [`in_steps`](CubeStagedMicrokernelSettings::in_steps) takes them: the float staged routine's. A
    /// cube decodes every block of the rhs it stages once, for all its rows, so the boxes more
    /// planes high give that decode more rows to serve, and the box four steps wide along `N`
    /// serves each row of the lhs it stages to more columns.
    const PACKED_BOXES: [((usize, usize), (usize, usize)); 9] = [
        ((4, 2), (2, 4)),
        ((4, 4), (2, 2)),
        ((8, 1), (1, 2)),
        ((4, 2), (1, 2)),
        ((4, 2), (2, 2)),
        ((4, 2), (4, 1)),
        ((8, 2), (4, 1)),
        ((8, 1), (2, 2)),
        ((8, 1), (4, 2)),
    ];

    /// Boxes against a packed input one step of `M` and `N` a plane, which holds every row of a
    /// product of a handful of rows, on planes side by side along `N`. A box no taller than the
    /// rows leaves no instruction idle, and a narrow one puts many cubes on the rhs, which so
    /// few rows stream once.
    const PACKED_ONE_STEP_BOXES: [((usize, usize), (usize, usize)); 3] =
        [((1, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 1), (1, 4))];

    /// Planes sharing a one-step box along `K`. One plane walking all of `K` stalls on each
    /// stage's fill, and a projection with few columns has too few cubes to hide it; past eight,
    /// a stage of `K` that many planes deep outgrows shared memory.
    const PACKED_ONE_STEP_PLANES_ALONG_K: [usize; 2] = [4, 8];

    /// The settings a product of a handful of rows against a packed input races: one step of `M`
    /// and `N` a plane, on one, two or four planes along `N`, split four or eight ways along `K`,
    /// one stage at a time or three in flight. Measured on Qwen3-4B's projections at one to sixteen
    /// rows on an RTX PRO 6000: these were the fastest of every box, split and ring tried. The
    /// first, one plane split four ways one stage at a time, is the lead a caller that measures
    /// nothing takes.
    pub fn packed_one_step_settings() -> Vec<CubeStagedMicrokernelSettings> {
        let mut settings = Vec::new();
        for ring in [StageRing::SINGLE, StageRing::TRIPLE] {
            for (steps, planes) in Self::PACKED_ONE_STEP_BOXES {
                for along_k in Self::PACKED_ONE_STEP_PLANES_ALONG_K {
                    settings.push(
                        CubeStagedMicrokernelSettings::in_steps(steps, planes, ring)
                            .split_along_k(along_k),
                    );
                }
            }
        }
        settings
    }

    /// The settings a product against a packed input races on `hw`: the lead, the boxes several
    /// steps tall the float staged routine races, one stage at a time and with a second in flight,
    /// then the one-step boxes ([`packed_one_step_settings`](Self::packed_one_step_settings)).
    /// Planes splitting `K` only on a device whose shared memory takes the atomic `f32` add they
    /// merge through: elsewhere they would only be tried and declined.
    ///
    /// None through the `mma` transport: on an L4 its boxes lead `cmma`'s on Qwen3-4B's down
    /// projection by a few percent, and lose a third once the projection's residual add is fused
    /// into its store, which its fragments drain into cell by cell; the race sees only the
    /// product.
    pub fn packed_settings(hw: &Hardware) -> Vec<CubeStagedMicrokernelSettings> {
        let boxed = [StageRing::SINGLE, StageRing::DOUBLE]
            .into_iter()
            .flat_map(|ring| {
                Self::PACKED_BOXES.map(|(steps, planes)| {
                    CubeStagedMicrokernelSettings::in_steps(steps, planes, ring)
                })
            });
        let mut settings = vec![CubeStagedMicrokernelSettings::PACKED];
        for candidate in boxed.chain(Self::packed_one_step_settings()) {
            let merges = candidate.planes_along_k.get() > 1;
            if (!merges || hw.shared_f32_atomic_add()) && !settings.contains(&candidate) {
                settings.push(candidate);
            }
        }
        settings
    }

    /// The packed inputs' scale block along `K`: a stage holds whole ones of each. `None` where a
    /// packed input's scales do not group `K`, or two packed inputs group it differently.
    pub fn block(item: &MatmulWorkItem) -> Option<usize> {
        let space = item.blocks?;
        let mut blocks = MatmulOperand::INPUTS
            .into_iter()
            .filter(|&operand| space.packings.get(operand).is_some())
            .map(|operand| {
                let packed = space.operand(operand);
                match packed.scaled_along() {
                    Some(axis) if axis == K => packed.grouping(K, space.k),
                    _ => None,
                }
            });
        let block = blocks.next()??;
        blocks.all(|other| other == Some(block)).then_some(block)
    }

    /// The item its planes contract, of `item`: `item` itself where every input is float; its
    /// packed inputs decoded where its fill decodes them into the stages
    /// ([`MatmulWorkItem::decoded`]); read as stored where it stages them as they lie
    /// ([`MatmulWorkItem::as_stored`]).
    pub fn contracted(&self, item: &MatmulWorkItem) -> MatmulWorkItem {
        match (item.blocks, self.walk().decoded()) {
            (None, _) => *item,
            (Some(_), Some(_)) => item.decoded(),
            (Some(_), None) => item
                .as_stored()
                .expect("a microkernel staging packed inputs as they lie reads them as stored"),
        }
    }

    /// The planes of an item it stages as it lies, a float one or one read as stored: every
    /// plane microkernel this device offers for `item`, at the settings' layout and requests.
    fn as_stored_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeStagedMicrokernelSettings,
    ) -> impl Iterator<Item = Self> {
        let requested = *settings;
        PlaneFragmentMicrokernel::candidates(hw, item, &settings.plane).map(move |plane| Self {
            plane,
            planes: requested.planes,
            planes_along_k: requested.planes_along_k,
            scratch: requested.scratch,
            ring: requested.ring,
            fill: requested.fill,
            stage_rows: requested.stage_rows,
            store: requested.store,
            depth: CubeStagedDepth::Steps,
            reads: MatmulOperands::all(MatmulRead::AsBound),
        })
    }

    /// The planes over the block-split space of `item`: first those reading its packed inputs as
    /// stored ([`MatmulWorkItem::as_stored`]), where the device has an instruction over their
    /// values, then those on the stages they are decoded into
    /// ([`decoding_candidates`](Self::decoding_candidates)).
    fn candidates_in_blocks(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeStagedMicrokernelSettings,
    ) -> Vec<Self> {
        let mut candidates: Vec<Self> = match item.as_stored() {
            Some(stored) => Self::as_stored_candidates(hw, &stored, settings).collect(),
            None => Vec::new(),
        };
        candidates.extend(Self::decoding_candidates(hw, item, settings));
        candidates
    }

    /// The planes against the packed inputs of `item`, at `settings` counted in its steps: the
    /// float product's plane microkernels on the decoded stages, through the transport the
    /// settings ask for first, one for each matrix instruction under which a step of `K`, the
    /// stage, is whole scale blocks, each a whole number of the instruction's steps. A plane holds the settings' steps of `M` and `N` and walks a step of
    /// `K` through each stage, block by block, in the instruction's calls ([`MatmulStep::calls`]).
    /// Only the matrix instructions: the software one's units read the stage at widths this
    /// routine does not bind. The units filling each stage decode every packed input into
    /// [`MatmulWorkItem::decoded_element`].
    fn decoding_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeStagedMicrokernelSettings,
    ) -> Vec<Self> {
        let (block, read) = (Self::block(item), item.decoded_element());
        let step = item
            .blocks
            .map(|space| MatmulStep::new(hw, space.operands(), MatmulOperands::all(read)));
        PlaneFragmentMicrokernel::candidates(
            hw,
            &item.decoded(),
            &PlaneFragmentMicrokernelSettings {
                transport: settings.plane.transport,
                ..PlaneFragmentMicrokernelSettings::DEFAULT
            },
        )
        .filter(|plane| {
            !matches!(
                plane.instruction,
                PlaneFragmentInstruction::Registers { .. }
            )
        })
        .filter_map(|plane| {
            let [(_, m), (_, n), (_, k)] = plane.instruction.shape();
            let step = step.clone()?.under(Space::new(&[(M, m), (N, n), (K, k)]));
            let block = block?;
            if !step.along(K).is_multiple_of(block) || !block.is_multiple_of(k) {
                return None;
            }
            let blocks = step.along(K) / block;
            let plane = PlaneFragmentMicrokernel {
                settings: PlaneFragmentMicrokernelSettings {
                    along_m: settings
                        .plane
                        .along_m
                        .checked_mul(NonZeroUsize::new(step.calls(M))?)?,
                    along_n: settings
                        .plane
                        .along_n
                        .checked_mul(NonZeroUsize::new(step.calls(N))?)?,
                    depth: NonZeroUsize::new(step.calls(K) / blocks)?,
                    ..plane.settings
                },
                ..plane
            };
            Some(Self {
                plane,
                planes: settings.planes,
                planes_along_k: settings.planes_along_k,
                scratch: settings.scratch,
                ring: settings.ring,
                fill: settings.fill,
                stage_rows: settings.stage_rows,
                store: settings.store,
                depth: CubeStagedDepth::ScaleBlocks(NonZeroUsize::new(blocks)?),
                reads: item.decoding_reads(),
            })
        })
        .collect()
    }

    /// How its planes reach its output: each its own tile, or merged along `K`.
    pub fn drain(&self) -> CubeStagedDrain {
        match self.planes_along_k.get() {
            1 => CubeStagedDrain::EachPlane,
            _ => CubeStagedDrain::Merged,
        }
    }

    /// What the cube's step reads of this microkernel.
    pub fn walk(&self) -> CubeStagedWalk {
        CubeStagedWalk {
            drain: self.drain(),
            filling_planes: self.fill.filling_planes,
            ring: self.ring,
            scratch: self.scratch,
            instruction: self.plane.tile_instruction(),
            step_sum: self.plane.instruction.step_sum(),
            store: self.store,
            reads: self.reads,
        }
    }

    /// Planes one cube holds that contract: its layout along `m` and `n` times the planes along
    /// `K`, each a group of the planes its instruction spans.
    fn contracting_planes(&self) -> usize {
        self.planes.count() * self.planes_along_k.get() * self.plane.instruction.planes()
    }

    /// The levels inside one cube's box, which a routine places the cubes around: the plane's own,
    /// distributed over the cube's planes along its rows and columns and, where they split it, the
    /// contraction, walked through all of it a stage at a time. Every box and stage it gives is
    /// the cube's, whatever its fill turns out to set aside.
    ///
    /// A stage of scale blocks walks `KB`, its planes' steps inside each block
    /// ([`PlaneFragmentMicrokernel::tiling_over`]), and its planes split `KB`.
    ///
    /// An instruction several planes issue together, a warpgroup MMA's, distributes the box over
    /// groups of them instead: the layout counts groups.
    pub fn tiling(&self) -> Levels {
        let MatmulAxes { m, n, .. } = self.plane.axes;
        let (tiling, walked) = match self.depth {
            CubeStagedDepth::Steps => (self.plane.tiling(), self.plane.walked()),
            CubeStagedDepth::ScaleBlocks(blocks) => {
                (self.plane.tiling_over((KB, blocks.get())), KB)
            }
        };
        let mut planes = vec![
            (m, self.planes.along_m.get()),
            (n, self.planes.along_n.get()),
        ];
        if self.planes_along_k.get() > 1 {
            planes.push((walked, self.planes_along_k.get()));
        }
        let tiling = match self.plane.instruction.planes() {
            1 => tiling.planes(&planes),
            group => tiling.plane_groups(group, &planes),
        };
        tiling.walk_every(&[walked])
    }

    /// The levels a cube launches: its [`tiling`](Self::tiling), beside the planes its fill sets
    /// aside to fill the stages. Those take no tile of any level, so they change no box or stage
    /// the tiling's placement gave; launched over its tiling instead, a cube would hold too few
    /// planes for its fill.
    pub fn launch_tiling(&self) -> Levels {
        self.tiling().filled_by(self.fill.filling_planes)
    }

    /// This microkernel as `hw` runs it for `item` over `buffers`, placed in `partitioning` (its
    /// [`tiling`](Self::tiling) placed over the problem), its cubes reaching the output as
    /// `write`: its ring, scratch and fill decided, how its stages hold their cells, the width of
    /// its planes and the shared memory a cube holds. The routine launches it over its
    /// [`launch_tiling`](Self::launch_tiling).
    ///
    /// Its fragments bounce through a scratch wherever they cannot land their cells on their own:
    /// into an output they add into, where the planes merge along `K` or the cubes relay the
    /// output, and into one they cannot address whole, a tile the output's edge cuts short or a
    /// sink. The software instruction's cells and a manual fragment's land on their own.
    ///
    /// Whether this problem's edge cuts a tile short reads the exact `m`, which a tune key
    /// anchors: the problem a candidate is ranked on and the one it launches on can differ on it
    /// alone, and what the tuner ranked has to be what launches. So what resolving decides beyond
    /// the bounce itself reads the window wherever its fragments *can* bounce: the shared memory
    /// it holds charges that window, and planes are set aside to fill a stage only where no
    /// fragment can bounce.
    ///
    /// # Errors
    ///
    /// A matrix instruction on a device that commits to no plane width.
    pub fn resolve(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        buffers: &MatmulBuffers,
        write: MatmulCubeWrite,
        partitioning: &MatmulPartitioning,
    ) -> Result<CubeStagedResolved, Refusal> {
        assert_eq!(
            partitioning.as_tile().fillers(),
            0,
            "a cube resolves against its tiling's placement, which sets no plane aside"
        );
        let stage = partitioning.stage();
        let plane_width = self.plane.instruction.units(hw)?;
        let units = self.contracting_planes() * plane_width as usize;
        let ring = self
            .ring
            .resolve(hw, &stage.operands(item.elements, self.plane.orders), units);
        let instruction = self.plane.instruction;
        let landed = self.plane.landed_stage_storage(&stage);
        let merges = self.drain() == CubeStagedDrain::Merged || write.accumulates();
        let store = self
            .store
            .resolve(hw, item, buffers, partitioning, write, self.planes_along_k);
        // A box the TMA engine stores is drained whole into shared memory first, and the
        // descriptor clips it at the output's edge: no fragment's window is cut short there.
        let cut_short = partitioning.overhangs_output() && store == CubeStagedStore::Fragments;
        let window = match self.plane.instruction.drain() {
            PlaneFragmentDrain::Bounce => {
                let bouncing = true;
                self.scratch.at_least_bouncing(bouncing)
            }
            PlaneFragmentDrain::CellByCell | PlaneFragmentDrain::InLines => Scratch::None,
        };
        let bounces = merges || cut_short || item.destination == MatmulDestination::Sink;
        let scratch = match bounces {
            true => window,
            false => Scratch::None,
        };
        let fill = self.fill.resolve(
            hw,
            item,
            buffers,
            partitioning,
            &stage,
            &landed,
            &ring,
            &self.plane,
            window,
            self.planes_along_k,
        );
        if let PlaneFragmentInstruction::Wgmma { .. } = instruction
            && let Some(refusal) = CubeStagedWgmmaRefusal::new(&fill, self.planes_along_k, merges)
        {
            return Err(refusal.into());
        }
        // A box lands a stage as one run of whole rows; the units copy it into the instruction's
        // own blocks, their rows laid as the setting says. A layout that lays nothing differently,
        // a box's landing or the software instruction's strided stage, is no decision of its own:
        // resolved as spread, its launch is its spread twin's, and a table sees one launch rather
        // than timing it twice.
        let (storage, stage_rows) = match fill.delivery {
            CubeStagedDelivery::Tma => (landed, CubeStagedRows::Spread),
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => {
                let spread = self.plane.stage_storage(&stage);
                let laid = self.stage_rows.lay(spread.clone());
                let stage_rows = match laid == spread {
                    true => CubeStagedRows::Spread,
                    false => self.stage_rows,
                };
                (laid, stage_rows)
            }
        };
        let microkernel = Self {
            scratch,
            ring,
            fill,
            stage_rows,
            store,
            ..self
        };
        let shared_memory = microkernel.shared_memory(hw, item, write, &stage, &storage, window);
        Ok(CubeStagedResolved {
            microkernel,
            storage,
            plane_width,
            shared_memory,
        })
    }

    /// Bytes of shared memory one cube holds staging `stage` on `hw`: its ring's stages of both
    /// operands, laid out as `storage`, with the words of block scales a block-scaled
    /// instruction's stages keep beside them, or the `window` each contracting plane's fragments
    /// bounce through once the stages are freed, whichever is larger; where its planes merge along
    /// `K`, the accumulator over the box, one sum a cell; and where the TMA engine stores the box,
    /// the box it is drained into. The accumulator and the box are opened before the walk, so
    /// they sit beside the stages.
    pub(crate) fn shared_memory(
        &self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        write: MatmulCubeWrite,
        stage: &MatmulStage,
        storage: &StageStorage,
        window: Scratch,
    ) -> usize {
        let accumulator = match self.drain() {
            CubeStagedDrain::EachPlane => 0,
            CubeStagedDrain::Merged => {
                stage
                    .operand_space(MatmulOperand::Out, self.plane.orders)
                    .cells()
                    * item.sum.size()
            }
        };
        let windows = self.contracting_planes() * self.plane.scratch_bytes(item, window);
        let scales = self.scale_bytes(stage);
        let stored = self
            .store
            .box_bytes(stage, item.elements.get(MatmulOperand::Out).size());
        let stages = self
            .ring
            .bytes(stage.bytes(hw, item.elements, storage, self.plane.orders) + scales);
        // The windows open once the walk has freed its stages, and take their bytes
        // (`CubeStagedWalk::drain_sum`), where a cube walks its run once; one that opens stages for
        // each box of its run, in a loop, frees nothing the windows could take.
        let walked = match write.walks_once() {
            true => stages.max(windows),
            false => stages + windows,
        };
        walked + accumulator + stored
    }

    /// Bytes one stage's block scales take beside its values, where the instruction is
    /// block-scaled: a scale a row of each operand and a block of `K`, one byte each. Beside a box
    /// the TMA engine lands, each operand's box may also sit up to its alignment past the scale
    /// words staged before it.
    fn scale_bytes(&self, stage: &MatmulStage) -> usize {
        if !self.plane.instruction.block_scaled() {
            return 0;
        }
        let axes = stage.axes();
        let scales = (stage.extent(axes.m) + stage.extent(axes.n)) * stage.extent(KB);
        let landing = match self.fill.delivery {
            CubeStagedDelivery::Tma => 2 * Delivery::TMA_STAGE_ALIGNMENT,
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => 0,
        };
        scales + landing
    }

    /// Panics where `out`'s writes are masked but its fragments hold no scratch to bounce
    /// through. The engine masks the output's writes exactly where its tiles overhang, and a
    /// masked write is one a fragment's store intrinsic cannot make.
    pub fn assert_drains_into(&self, out: &Bound) {
        assert!(
            self.plane.instruction.drain() != PlaneFragmentDrain::Bounce
                || self.scratch != Scratch::None
                || !out.spec.is_checked(),
            "the output's writes are masked, but its fragments were given no scratch to bounce \
             through"
        );
    }
}

/// A cube microkernel resolved against its box ([`CubeStagedMicrokernel::resolve`]), and what
/// resolving decided: how its stages hold their cells, the width of its planes, and the bytes of
/// shared memory one cube holds.
#[derive(Debug, Clone)]
pub struct CubeStagedResolved {
    pub microkernel: CubeStagedMicrokernel,
    pub storage: StageStorage,
    pub plane_width: u32,
    pub shared_memory: usize,
}

impl Microkernel for CubeStagedMicrokernel {
    type WorkItem = MatmulWorkItem;
    type Settings = CubeStagedMicrokernelSettings;
    type Operand = MatmulOperand;
    type Partitioning = MatmulPartitioning;

    const DECLINE: &'static str = "this device runs no plane of these settings: planes along K merge through an atomic f32 add in shared memory, and against a packed input a plane runs a matrix instruction the device offers for the float input's element, at a plane width it commits to, whose step of K is whole scale blocks, each a whole number of the instruction's steps";

    /// A star around the default, one axis a setting names at a time: the fragment grid, the plane
    /// layout. Then the named boxes a star moving one axis never reaches — the deep ones, two
    /// instruction steps of `K` a stage, which read longer runs of a col rhs's rows; the small
    /// ones a short problem fills the device with; the small cubes, one to four planes on a line
    /// each holding a deep grid, where a small instruction (Metal's `8×8×8` simdgroup matrix, or
    /// the software instruction) contracts more per shared read and more cubes fit a core; the tile
    /// DSL's multi-row box ([`MULTI_ROW`](CubeStagedMicrokernelSettings::MULTI_ROW)) — the named
    /// ones crossed with the rings that keep a second stage in flight and the transport that
    /// reaches its instruction.
    ///
    /// Then the default's grid on two by two planes and the small cube four fragments deep on two
    /// planes, each with its stage's rows in order ([`CubeStagedRows::InOrder`]): a col rhs's
    /// fragments load down `K`, where the padding that spreads a row rhs's loads across banks
    /// can cost more than the conflicts it saves.
    ///
    /// Then the planes splitting `K`
    /// ([`SPLIT_ALONG_K`](CubeStagedMicrokernelSettings::SPLIT_ALONG_K)), then two and eight of
    /// them, the whole partition resident in the scratch, a second stage in flight, filled in its
    /// slot and through registers, the latter on two planes too.
    ///
    /// Then, on a device with a copy engine, the lead, the named boxes and the rings that fill in
    /// their slots again, their stages handed to the engine ([`CubeStagedDelivery::AsyncCopy`]):
    /// whether it beats the units' loads moves with the box, by a few percent either way on an
    /// L4. A ring that prefetches through registers is not listed with it, which it refuses.
    ///
    /// Then, on a device with a bulk tensor copy, boxes whose stages the TMA engine fills
    /// ([`CubeStagedDelivery::Tma`]) through the tile DSL's transport, two and three stages in
    /// their slots, some beside a plane set aside to fill them, one whose box the TMA engine stores
    /// too ([`CubeStagedStore::Tma`]). On a device with a warpgroup MMA too, two plane groups along
    /// `m`, each one `64 × 128` instruction four steps of `K` a stage (a 128-byte swizzle span of
    /// `f16`), beside the plane that fills the stages, which a warpgroup MMA needs to release a
    /// stage only once its MMAs are done with it.
    ///
    /// Never the narrow step, which changes what the product computes, a thing a race on time
    /// alone cannot weigh
    /// ([`PlaneFragmentStepSum::Narrow`](crate::matmul::dense::plane::PlaneFragmentStepSum::Narrow)).
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        use PlaneFragmentMicrokernelSettings as Plane;
        use PlaneFragmentTransport::{Mma, Wgmma};
        let lead = Self::Settings::DEFAULT;
        let count = |n: usize| NonZeroUsize::new(n).unwrap();
        let layout = MatmulPlaneLayout::new;
        let boxed = |(rows, columns, depth), (plane_rows, plane_columns)| {
            Self::Settings::boxed(
                Plane::grid(rows, columns, depth),
                layout(plane_rows, plane_columns),
            )
        };

        let grids = [(4, 4, 2), (2, 4, 1), (4, 2, 1), (2, 2, 1)]
            .map(|grid| boxed(grid, (lead.planes.along_m.get(), lead.planes.along_n.get())));
        let layouts = [(2, 2), (4, 2), (4, 4)].map(|planes| boxed((4, 4, 1), planes));

        let deep = grids[0];
        let deep_on_2x2 = boxed((4, 4, 2), (2, 2));
        let deep_on_4x2 = boxed((4, 4, 2), (4, 2));
        let small_on_2x1 = boxed((2, 4, 2), (2, 1));
        let small_on_2x2 = boxed((2, 4, 2), (2, 2));
        let multi_row = Self::Settings::MULTI_ROW;
        let small_cubes = [
            ((4, 4, 2), (1, 2)),
            ((8, 2, 2), (1, 2)),
            ((2, 4, 4), (2, 1)),
            ((4, 4, 2), (4, 1)),
            ((8, 4, 2), (4, 1)),
        ]
        .map(|(grid, planes)| boxed(grid, planes));
        let in_order = |settings: Self::Settings| Self::Settings {
            stage_rows: CubeStagedRows::InOrder,
            ..settings
        };
        let rows_in_order = [layouts[0], small_cubes[2]].map(in_order);
        let boxes = [
            deep_on_2x2,
            deep_on_4x2,
            small_on_2x1,
            small_on_2x2,
            multi_row,
        ];

        let double = StageRing::DOUBLE;
        let prefetching = StageRing::DOUBLE_THROUGH_REGISTERS;
        let prefetching_one_stage = StageRing::SINGLE_THROUGH_REGISTERS;
        // A deep box on eight planes stages 40 KB padded, so its second stage is past a core of
        // 64 KB of shared memory, and it is listed with its next stage waiting in registers alone
        // too: whichever of the two the device's shared memory refuses is not raced.
        let rings = [
            lead.ringed(double),
            multi_row.ringed(double),
            lead.ringed(prefetching_one_stage),
            lead.ringed(prefetching),
            multi_row.ringed(prefetching),
            deep.ringed(prefetching),
            deep.ringed(prefetching_one_stage),
            deep_on_2x2.ringed(prefetching),
            deep_on_4x2.ringed(prefetching),
            deep_on_4x2.ringed(prefetching_one_stage),
            small_on_2x1.ringed(prefetching),
            small_on_2x2.ringed(prefetching),
        ];
        // The deep box in the `mma` instruction's terms: its `16 × 8` fragments two by sixteen
        // give the plane the `32 × 128` tile four by four of `cmma`'s `8 × 32` do. The deep box's
        // own fragments give it `64 × 32`, a box too narrow along `n` to reuse its lhs.
        let deep_for_mma = boxed((2, 16, 2), (4, 1));
        let transports = [
            deep_for_mma,
            deep_for_mma.ringed(prefetching),
            deep_for_mma.ringed(prefetching_one_stage),
            deep,
            deep.ringed(prefetching),
            deep.ringed(prefetching_one_stage),
            small_on_2x2,
            small_on_2x1.ringed(prefetching),
            small_on_2x2.ringed(prefetching),
        ]
        .map(|settings| settings.through(Mma));

        let split_lead = Self::Settings::SPLIT_ALONG_K;
        let split = |planes_along_k| Self::Settings {
            planes_along_k: count(planes_along_k),
            ..split_lead
        };
        let splits = [
            split(2),
            split(8),
            Self::Settings {
                scratch: Scratch::WholeGrid,
                ..split_lead
            },
            split_lead.ringed(double),
            split_lead.ringed(prefetching),
            split(2).ringed(prefetching),
        ];

        let engine = hw.unit_async_copy().then(|| {
            [
                lead,
                deep,
                deep_on_2x2,
                deep_on_4x2,
                small_on_2x1,
                small_on_2x2,
                multi_row,
                deep.through(Mma),
                small_on_2x2.through(Mma),
                lead.ringed(double),
                multi_row.ringed(double),
            ]
            .map(|settings| {
                let fill = settings.fill;
                settings.filled(CubeStagedDelivery::AsyncCopy, fill.filling_planes)
            })
        });

        // Through the tile DSL's transport, whose `ldmatrix` reads the rows a box lands swizzled,
        // in their slots, which is the only ring a bulk copy fills. A ring deeper than two keeps
        // the engine more than one stage ahead, which a two-stage ring cannot: its next fill waits
        // on every plane having read the stage before. The boxes are the ones that won on the Qwen3
        // projections at 512 and 2048 rows on a 188-core Blackwell, crossed with the ring, the
        // filling plane and the store each won with: the `128 × 128` deep box, and the same on two
        // by two planes of `64 × 64`, whose fragments reuse each operand load twice as much; the
        // `64 × 128` boxes beside a filling plane, which put twice the cubes on a narrow projection
        // a prompt's chunk of rows would leave cores idle under, the deeper one stored by the TMA
        // engine, whose box costs no cube a core where the stages already hold one alone; the `256
        // × 128` and `128 × 256` boxes, on eight planes, for the long products; and the `64 × 64`
        // box beside a filling plane, stored by the TMA engine, which a projection with few
        // columns needs to fill the device at all: a 512-row chunk of Qwen3-0.6B's output projection
        // is 128 of the `64 × 128` boxes for 188 cores, and on the `64 × 64` it takes 16.4 µs
        // against 22.5, what cuBLAS takes; and the `128 × 128` box on two by four planes, each
        // four by four fragments four steps deep, beside a filling plane, which the long
        // projections of Qwen3-8B want: its output projection takes 73.7 µs against the deep box's
        // 86.0, its down projection 184.3 against 227.3. Every one of them fits two stages in
        // 64 KB and its own ring and box in 99 KB.
        let filled_by_tma = |grid, planes, ring, filling_planes| {
            boxed(grid, planes)
                .ringed(ring)
                .through(Mma)
                .filled(CubeStagedDelivery::Tma, filling_planes)
        };
        let stored_by_tma = |settings: Self::Settings| Self::Settings {
            store: CubeStagedStore::Tma,
            ..settings
        };
        let tma = hw.bulk_tensor_copy().then(|| {
            [
                filled_by_tma((2, 16, 2), (4, 1), double, 0),
                filled_by_tma((4, 8, 2), (2, 2), StageRing::TRIPLE, 0),
                stored_by_tma(filled_by_tma((2, 8, 4), (2, 2), StageRing::TRIPLE, 1)),
                filled_by_tma((2, 8, 2), (2, 2), double, 1),
                filled_by_tma((2, 16, 2), (8, 1), double, 0),
                filled_by_tma((2, 16, 2), (4, 2), StageRing::TRIPLE, 0),
                stored_by_tma(filled_by_tma((2, 4, 4), (2, 2), StageRing::TRIPLE, 1)),
                filled_by_tma((4, 4, 4), (2, 4), StageRing::TRIPLE, 1),
            ]
        });
        let wgmma = (hw.bulk_tensor_copy()
            && hw.properties.features.matmul.wgmma.iter().next().is_some())
        .then(|| {
            boxed((1, 16, 4), (2, 1))
                .ringed(double)
                .filled(CubeStagedDelivery::Tma, 1)
                .through(Wgmma)
        });

        [lead]
            .into_iter()
            .chain(grids)
            .chain(layouts)
            .chain(boxes)
            .chain(small_cubes)
            .chain(rings)
            .chain(transports)
            .chain(rows_in_order)
            .chain([split_lead])
            .chain(splits)
            .chain(engine.into_iter().flatten())
            .chain(tma.into_iter().flatten())
            .chain(wgmma)
            .collect()
    }

    /// Items on matmul's own axes, which it places and stages in; a packed one, against a packed
    /// input whose scales group `K`, which it stages a block at a time.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.blocks {
            None => MatmulPartitioning::admits(item),
            Some(_) => Self::block(item).map(|_| ()).ok_or(Self::UNGROUPED_K),
        }
    }

    /// Where the settings ask for a warpgroup MMA, that its rules do not hold for this device or
    /// product ([`PlaneFragmentTransport::Wgmma`]); [`DECLINE`](Self::DECLINE) otherwise.
    fn decline(settings: &Self::Settings) -> &'static str {
        match settings.plane.transport {
            PlaneFragmentTransport::Wgmma => {
                "this setting asks for a warpgroup MMA, which this device or product does not take"
            }
            PlaneFragmentTransport::Cmma | PlaneFragmentTransport::Mma => Self::DECLINE,
        }
    }

    /// Every plane microkernel this device offers, at the settings' layout and requests; against
    /// packed inputs, those over their block-split space
    /// (`candidates_in_blocks`). None, where the planes split `K`,
    /// on a device whose shared memory takes no atomic `f32` add, which is where they merge.
    fn candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let merges = settings.planes_along_k.get() > 1;
        let atomic = hw.shared_f32_atomic_add();
        let candidates: Vec<Self> = match item.blocks {
            None => Self::as_stored_candidates(hw, item, settings).collect(),
            Some(_) => Self::candidates_in_blocks(hw, item, settings),
        };
        candidates.into_iter().filter(move |_| !merges || atomic)
    }
    /// Its ring reads both operands out of their tiles a stage at a time, in stretches of the
    /// stage's extent along each one's contiguous axis, one element wide where the TMA engine
    /// fills it. Each plane drains its tile a fragment at a time, or a cell at a time from a manual
    /// fragment, into the output or the carry the routine's cubes relay it through; merged along
    /// `K` in shared memory, it drains one element at a time, since an atomic takes one, and
    /// planes on the software instruction drain their register block, which holds the rhs's lines,
    /// so the rhs is read one element at a time too. Its planes' reads are the plane
    /// microkernel's.
    fn vectorize(
        &self,
        partitioning: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        assert_eq!(
            partitioning.as_tile().fillers() as usize,
            self.fill.filling_planes,
            "a cube is launched over its launch tiling, beside the planes its fill sets aside"
        );
        partitioning
            .stage()
            .require_runs(self.plane.orders, vectorization);
        self.fill.vectorize(vectorization);
        self.store.vectorize(vectorization);
        match self.drain() {
            CubeStagedDrain::EachPlane => {
                vectorization.require(VectorRequirement::Access {
                    operand: MatmulOperand::Out,
                    elements: partitioning.leaf().extent(
                        MatmulAxes::MNK.of(self.plane.orders.innermost(MatmulOperand::Out)),
                    ),
                });
                if self.plane.instruction.drain() == PlaneFragmentDrain::CellByCell {
                    vectorization.require(VectorRequirement::Elementwise {
                        operand: MatmulOperand::Out,
                    });
                }
            }
            CubeStagedDrain::Merged => {
                vectorization.require(VectorRequirement::Elementwise {
                    operand: MatmulOperand::Out,
                });
                if self.plane.instruction.drain() == PlaneFragmentDrain::InLines {
                    vectorization.require(VectorRequirement::Elementwise {
                        operand: MatmulOperand::Rhs,
                    });
                }
            }
        }
        self.plane.vectorize(&(), vectorization);
    }

    /// Its ring, staging a step of `K` at a time for all its planes: rows of the lhs contiguous
    /// along the whole of `K` a cube walks, of the rhs along the stage's `n`, or along the whole of
    /// `K` for a col rhs bound in its buffer's order. Its drain: each plane writing its tile a
    /// row of it at a time, or every plane adding its whole partial box into the accumulator
    /// atomically before the cube copies it out a row at a time; into an output the cubes relay,
    /// each turn reads the carry back. Then its planes' microkernel, for the work.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        partitioning: &MatmulPartitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let extents = &partitioning.cube_box();
        let stage = partitioning.stage();
        let (out_run, drain_atomics) = match self.drain() {
            CubeStagedDrain::EachPlane => (self.plane.tile_extent(N), 0),
            CubeStagedDrain::Merged => (
                extents.extent(N),
                (self.planes_along_k.get() * extents.extent(M) * extents.extent(N)) as u64,
            ),
        };
        let staging = MatmulStagingCount {
            drain_atomics,
            ..MatmulStagingCount::new(partitioning, &stage, self.plane.orders, &self.ring, out_run)
        };
        staging.count(item, partitioning, estimate);
        self.plane.estimate(item, &(), planes, estimate);
    }
}

/// Its planes' microkernel, and the scale blocks a stage holds where it holds them.
impl core::fmt::Display for CubeStagedMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.plane.fmt(f)?;
        match self.depth {
            CubeStagedDepth::Steps => Ok(()),
            CubeStagedDepth::ScaleBlocks(blocks) => write!(f, " blocks={blocks}"),
        }
    }
}

impl MatmulMicrokernel for CubeStagedMicrokernel {
    /// A cube owns a box of the rows.
    const ROWS_KIND: MatmulRowsKind = MatmulRowsKind::Tiled;

    /// Its planes' order: they read the rhs's tiles out of its stages.
    fn orders(&self) -> MatmulOperands<MatmulOrder> {
        self.plane.orders
    }

    fn delivery(&self) -> Delivery {
        self.fill.delivery.bound()
    }

    /// As their words where its planes' instruction is block-scaled, which takes a word of them as
    /// its scale register.
    fn block_scales(&self) -> BlockScales {
        match self.plane.instruction.block_scaled() {
            true => BlockScales::Words,
            false => BlockScales::Fields,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_std::devices;

    /// Planes splitting `K` against a packed input merge through an atomic `f32` add in shared
    /// memory, so a device without one lists none of them: a listed setting it cannot run would be
    /// tried and declined, and the merge without the atomic loses partial sums.
    #[test]
    fn packed_planes_split_along_k_only_where_shared_memory_adds_atomically() {
        for hw in [
            devices::strix_halo(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::cpu_12(),
        ] {
            let splits = CubeStagedMicrokernel::packed_settings(&hw)
                .into_iter()
                .any(|settings| settings.planes_along_k.get() > 1);
            assert_eq!(splits, hw.shared_f32_atomic_add(), "{hw:?}");
        }
    }
}
