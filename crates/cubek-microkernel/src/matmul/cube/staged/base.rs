use crate::matmul::cube::{
    CubeStagedDelivery, CubeStagedFill, CubeStagedFillingRefusal, CubeStagedMicrokernelSettings,
    CubeStagedStore, CubeStagedWalk, CubeStagedWgmmaRefusal, MatmulCubeSpace, MatmulPlaneLayout,
};
use crate::matmul::plane::{
    PlaneFragmentDrain, PlaneFragmentInstruction, PlaneFragmentMicrokernel,
    PlaneFragmentMicrokernelSettings,
};
use crate::matmul::{
    MatmulAxes, MatmulDestination, MatmulDim, MatmulMicrokernel, MatmulOperand, MatmulOperands,
    MatmulRead, MatmulStage, MatmulStagingCount, MatmulStep, MatmulWorkItem,
};
use core::num::NonZeroUsize;
use cubecl::ir::ElemType;
use cubecl::tile::launch::{Delivery, Refusal, VectorRequirement, Vectorization};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::space::Coverage;
use cubecl::tile::{Levels, Scratch, Space, StageStorage};
use cubek_std::Hardware;
use cubek_std::layout::BindingOrder;
use cubek_std::layout::stored::BlockScales;
use cubek_std::matmul_axes::{K, KB, M, N};
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel, StageLayout, StageRing};

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
/// settled. The scratch and the ring are the settings' requests until it is resolved against the
/// box a routine places it in ([`resolve`](Self::resolve)). Who fills its stages and who stores
/// its box are the routine's structure: [`candidates`](Microkernel::candidates) fill by the units
/// and store by the fragments, and a routine of another engine sets its own.
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
    /// Who fills them: the routine's.
    pub fill: CubeStagedFill,
    /// How a stage lays out the rows its fragments load: the layout its units copy it in, or
    /// none where a box lands it.
    pub stage_layout: Option<StageLayout>,
    /// Who stores its box of the output: the routine's.
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
        why: "the packed staged matmul stages K a scale block at a time, and this packed input's scales do not group K",
    };

    /// The microkernel `settings` ask for, its planes running `plane`, a stage spanning `depth`
    /// along the contraction, each operand read as `reads` says.
    fn new(
        settings: &CubeStagedMicrokernelSettings,
        plane: PlaneFragmentMicrokernel,
        depth: CubeStagedDepth,
        reads: MatmulOperands<MatmulRead>,
    ) -> Self {
        Self {
            plane,
            planes: settings.planes,
            planes_along_k: settings.planes_along_k,
            scratch: settings.scratch,
            ring: settings.ring,
            fill: CubeStagedFill::BY_UNITS,
            stage_layout: settings.stage_layout,
            store: CubeStagedStore::Fragments,
            depth,
            reads,
        }
    }

    /// The packed inputs' scale block along `K`: a stage holds whole ones of each. `None` where a
    /// packed input's scales do not group `K`, or two packed inputs group it differently.
    pub(crate) fn block(item: &MatmulWorkItem) -> Option<usize> {
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

    /// The item its planes contract, of `item`: `item` itself where every input is float, or its
    /// inputs read at the element its fill rounds them to; its packed inputs decoded where its
    /// fill decodes them into the stages
    /// ([`MatmulWorkItem::decoded`]); read as stored where it stages them as they lie
    /// ([`MatmulWorkItem::as_stored`]).
    pub fn contracted(&self, item: &MatmulWorkItem) -> MatmulWorkItem {
        match (item.blocks, self.walk().decoded()) {
            (None, None) => *item,
            (None, Some(read)) => item.read_as(read),
            (Some(_), Some(_)) => item.decoded(),
            (Some(_), None) => item
                .as_stored()
                .expect("a microkernel staging packed inputs as they lie reads them as stored"),
        }
    }

    /// The planes of an item it stages as it lies, a float one or one read as stored: where its
    /// inputs may be rounded ([`MatmulWorkItem::rounded`]), first every plane on a matrix
    /// instruction over the rounded inputs, which the units' fill decodes into its stages as it
    /// loads them; then every plane microkernel this device offers for `item`, at the settings'
    /// layout and requests.
    ///
    /// Rounded first, as the multi-level routines promote an `f32` product to tf32 where they
    /// may: on a device whose matrix instructions take no `f32`, no exact plane runs.
    fn as_stored_candidates(
        hw: &Hardware,
        item: &MatmulWorkItem,
        settings: &CubeStagedMicrokernelSettings,
    ) -> Vec<Self> {
        let mut candidates = Vec::new();
        if let Some(read) = item.rounded() {
            let reads = MatmulOperands {
                lhs: MatmulRead::Dequantized(read),
                rhs: MatmulRead::Dequantized(read),
                out: MatmulRead::AsBound,
            };
            candidates.extend(
                PlaneFragmentMicrokernel::candidates(hw, &item.read_as(read), &settings.plane)
                    .map(|plane| Self::new(settings, plane, CubeStagedDepth::Steps, reads)),
            );
        }
        // A packed rhs's plane reads the stage as the rhs is stored, `K` in its blocks.
        candidates.extend(
            PlaneFragmentMicrokernel::candidates(hw, item, &settings.plane)
                .map(|plane| match plane.blocks {
                    Some(_) => plane.reading_a_stage_as_stored(),
                    None => plane,
                })
                .map(|plane| {
                    Self::new(
                        settings,
                        plane,
                        CubeStagedDepth::Steps,
                        MatmulOperands::AS_BOUND,
                    )
                }),
        );
        candidates
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
            Some(stored) => Self::as_stored_candidates(hw, &stored, settings),
            None => Vec::new(),
        };
        candidates.extend(Self::decoding_candidates(hw, item, settings));
        candidates
    }

    /// The planes against the packed inputs of `item`, at `settings` counted in its steps: the
    /// float product's plane microkernels on the decoded stages, the tile layer's instruction
    /// first, one for each matrix instruction under which a step of `K`, the stage, is
    /// whole scale blocks, each a whole number of the instruction's steps. A plane holds the
    /// settings' steps of `M` and `N` and walks a step of `K` through each stage, block by block,
    /// in the instruction's calls ([`MatmulStep::calls`]). The units filling
    /// each stage decode every packed input into [`MatmulWorkItem::decoded_element`].
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
            &PlaneFragmentMicrokernelSettings::DEFAULT,
        )
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
            let depth = CubeStagedDepth::ScaleBlocks(NonZeroUsize::new(blocks)?);
            Some(Self::new(settings, plane, depth, item.decoding_reads()))
        })
        .collect()
    }

    /// How its planes reach its output: each its own tile, or merged along `K`.
    pub(crate) fn drain(&self) -> CubeStagedDrain {
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
            store: self.store,
            reads: self.reads,
            rows_along: self.plane.settings.rows_along,
        }
    }

    /// Planes one cube holds that contract: its layout along `m` and `n` times the planes along
    /// `K`, each a group of the planes its instruction spans.
    fn contracting_planes(&self) -> usize {
        self.planes.count() * self.planes_along_k.get() * self.plane.instruction.planes()
    }

    /// The box one step of its walk through `K` stages, in the space `cube_space` its cube covers:
    /// the first level of its [`tiling`](Self::tiling) that walks `K` (its blocks, in a
    /// block-split space), cut from that space.
    pub fn stage(&self, cube_space: &MatmulCubeSpace) -> MatmulStage {
        let walked = cube_space.axes_kind.walked();
        let mut enclosing = cube_space.space.clone();
        for level in self.tiling().build() {
            let tile = level.child(&enclosing);
            if level.coverage() == Coverage::Walk && level.tile(walked).is_some() {
                return MatmulStage::new(tile, cube_space.axes_kind.stage_axes());
            }
            enclosing = tile;
        }
        unreachable!("a staged cube's tiling walks `K`")
    }

    /// The levels a cube launches: its [`tiling`](Self::tiling), beside the planes its fill sets
    /// aside to fill the stages. Those take no tile of any level, so they change no box or stage
    /// the tiling's placement gave; launched over its tiling instead, a cube would hold too few
    /// planes for its fill.
    pub fn launch_tiling(&self) -> Levels {
        self.tiling().filled_by(self.fill.filling_planes)
    }

    /// This microkernel as `hw` runs it for `item`, handed the box `cube_space` (of its
    /// [`tiling`](Self::tiling) placed over the problem): its ring and scratch decided, how its
    /// stages hold their cells, the width of its planes and the shared memory a cube holds. The
    /// routine launches it over its [`launch_tiling`](Self::launch_tiling). Whether its fill and
    /// store engines serve the problem's buffers is the routine's to check.
    ///
    /// Its fragments bounce through a scratch wherever they cannot land their cells on their own:
    /// into an output they add into, where the planes merge along `K`, and into one they cannot
    /// address whole, a tile the output's edge cuts short or a sink. A manual fragment's cells
    /// land on their own.
    ///
    /// Whether this problem's edge cuts a tile short reads the exact `m`, which a tune key
    /// anchors: the problem a candidate is ranked on and the one it launches on can differ on it
    /// alone, and what the tuner ranked has to be what launches. So what resolving decides beyond
    /// the bounce itself reads the window wherever its fragments *can* bounce: the shared memory
    /// it holds charges that window, and planes set aside to fill a stage are refused wherever a
    /// fragment can bounce.
    ///
    /// # Errors
    ///
    /// A matrix instruction on a device that commits to no plane width; planes set aside to fill
    /// where they cannot be ([`CubeStagedFillingRefusal`]); a warpgroup MMA its fill and drain do
    /// not serve (`CubeStagedWgmmaRefusal`).
    pub fn resolve(
        self,
        hw: &Hardware,
        item: &MatmulWorkItem,
        cube_space: &MatmulCubeSpace,
    ) -> Result<CubeStagedResolved, Refusal> {
        let stage = &self.stage(cube_space);
        let plane_width = self.plane.instruction.units(hw)?;
        let units = self.contracting_planes() * plane_width as usize;
        let operands = stage.operands(item.elements, self.plane.orders, self.walk().reads);
        let ring = self.ring.resolve(hw, &operands, units);
        let instruction = self.plane.instruction;
        let landed = self.plane.landed_stage_storage(stage);
        let merges = self.drain() == CubeStagedDrain::Merged;
        // A box the TMA engine stores is drained whole into shared memory first, and the
        // descriptor clips it at the output's edge: no fragment's window is cut short there.
        let cut_short = cube_space.overhangs_output && self.store == CubeStagedStore::Fragments;
        let window = match self.plane.instruction.drain() {
            PlaneFragmentDrain::Bounce => self.scratch.at_least_bouncing(true),
            PlaneFragmentDrain::CellByCell => Scratch::None,
        };
        let bounces = merges || cut_short || item.destination == MatmulDestination::Sink;
        let scratch = match bounces {
            true => window,
            false => Scratch::None,
        };
        let fill = self.fill;
        if fill.filling_planes > 0
            && let Some(refusal) =
                CubeStagedFillingRefusal::new(fill.delivery, &ring, window, self.planes_along_k)
        {
            return Err(refusal.into());
        }
        if let PlaneFragmentInstruction::Wgmma { .. } = instruction
            && let Some(refusal) = CubeStagedWgmmaRefusal::new(&fill, self.planes_along_k, merges)
        {
            return Err(refusal.into());
        }
        // A box lands a stage as one run of whole rows; the units copy it in the layout the
        // setting names, or the instruction's own. Resolved to the layout it lays, a setting that
        // names the instruction's own launches as its twin does, and a table sees one launch
        // rather than timing it twice.
        let (storage, stage_layout) = match fill.delivery {
            CubeStagedDelivery::Tma => (landed, None),
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => {
                let layout = instruction.stage_layout(self.stage_layout)?;
                (self.plane.stage_storage(layout, stage), Some(layout))
            }
        };
        let microkernel = Self {
            scratch,
            ring,
            stage_layout,
            ..self
        };
        let shared_memory = microkernel.shared_memory(hw, item, stage, &storage, window);
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
    /// the box it is drained into. The accumulator is opened before the walk, so it sits beside
    /// the stages; so is the box, except beside planes set aside to fill, which open it once the
    /// walk has freed the ring.
    pub fn shared_memory(
        &self,
        hw: &Hardware,
        item: &MatmulWorkItem,
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
        let scales = self.scale_bytes(item, stage);
        let stored = self
            .store
            .box_bytes(stage, item.elements.get(MatmulOperand::Out).size());
        let stages = self
            .ring
            .bytes(stage.bytes(hw, item.elements, storage, self.plane.orders) + scales);
        // The windows open once the walk has freed its stages, and take their bytes
        // (`CubeStagedWalk::drain_sum`); so does the box planes set aside to fill drain into.
        if self.fill.filling_planes > 0 {
            stages.max(windows).max(stored) + accumulator
        } else {
            stages.max(windows) + accumulator + stored
        }
    }

    /// Bytes one stage's block scales take beside its values, where its planes read the packed
    /// inputs as stored (the block-scaled instruction, or fragments that decode as they load):
    /// a scale a row of each packed input and a block of `K`, one byte each. Beside a box the TMA
    /// engine lands, each operand's box may also sit up to its alignment past the scale words
    /// staged before it. None where the fill decodes the packed inputs into the stages.
    fn scale_bytes(&self, item: &MatmulWorkItem, stage: &MatmulStage) -> usize {
        let Some(blocks) = item.blocks else {
            return 0;
        };
        if self.walk().decoded().is_some() {
            return 0;
        }
        let axes = stage.axes();
        let rows: usize = MatmulOperand::INPUTS
            .into_iter()
            .filter(|&operand| blocks.packings.get(operand).is_some())
            .flat_map(|operand| operand.dims())
            .filter(|&dim| dim != MatmulDim::K)
            .map(|dim| stage.extent(axes.of(dim)))
            .sum();
        let scales = rows * stage.extent(KB);
        let landing = match self.fill.delivery {
            CubeStagedDelivery::Tma => 2 * Delivery::TMA_STAGE_ALIGNMENT,
            CubeStagedDelivery::Copy | CubeStagedDelivery::AsyncCopy => 0,
        };
        scales + landing
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
    type CubeSpace = MatmulCubeSpace;

    const DECLINE: &'static str = "this device runs no plane of these settings: planes along K merge through an atomic f32 add in shared memory, and against a packed input a plane runs a matrix instruction the device offers for the float input's element, at a plane width it commits to, whose step of K is whole scale blocks, each a whole number of the instruction's steps";

    /// A star around the default, one axis a setting names at a time: the fragment grid, the plane
    /// layout. Then the named boxes a star moving one axis never reaches — the deep ones, two
    /// instruction steps of `K` a stage, which read longer runs of a col rhs's rows; the small
    /// ones a short problem fills the device with; the small cubes, one to four planes on a line
    /// each holding a deep grid, where a small instruction (Metal's `8×8×8` simdgroup matrix)
    /// contracts more per shared read and more cubes fit a core — the named ones crossed with the
    /// rings that keep a second stage in flight. The instruction is the tile layer's
    /// ([`Instruction::for_accumulator`](cubecl::tile::Instruction::for_accumulator)), never a
    /// setting.
    ///
    /// Then the default's grid on two by two planes and the small cube four fragments deep on two
    /// planes, each with its stage's rows in order ([`StageLayout::Blocks`]): a col rhs's
    /// fragments load down `K`, where the padding that spreads a row rhs's loads across banks
    /// can cost more than the conflicts it saves.
    ///
    /// Then the planes splitting `K`
    /// ([`SPLIT_ALONG_K`](CubeStagedMicrokernelSettings::SPLIT_ALONG_K)), then two and eight of
    /// them, the whole partition resident in the scratch, a second stage in flight, filled in its
    /// slot and through registers, the latter on two planes too.
    ///
    /// Every one fills its stages by the cube's units; the routines of the other engines race
    /// their own lists.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        use PlaneFragmentMicrokernelSettings as Plane;
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
        let small_cubes = [
            ((4, 4, 2), (1, 2)),
            ((8, 2, 2), (1, 2)),
            ((2, 4, 4), (2, 1)),
            ((4, 4, 2), (4, 1)),
            ((8, 4, 2), (4, 1)),
        ]
        .map(|(grid, planes)| boxed(grid, planes));
        let in_order = |settings: Self::Settings| Self::Settings {
            stage_layout: Some(StageLayout::Blocks),
            ..settings
        };
        let rows_in_order = [layouts[0], small_cubes[2]].map(in_order);
        let boxes = [deep_on_2x2, deep_on_4x2, small_on_2x1, small_on_2x2];

        let double = StageRing::DOUBLE;
        let prefetching = StageRing::DOUBLE_THROUGH_REGISTERS;
        let prefetching_one_stage = StageRing::SINGLE_THROUGH_REGISTERS;
        // A deep box on eight planes stages 40 KB padded, so its second stage is past a core of
        // 64 KB of shared memory, and it is listed with its next stage waiting in registers alone
        // too: whichever of the two the device's shared memory refuses is not raced.
        let rings = [
            lead.ringed(double),
            lead.ringed(prefetching_one_stage),
            lead.ringed(prefetching),
            deep.ringed(prefetching),
            deep.ringed(prefetching_one_stage),
            deep_on_2x2.ringed(prefetching),
            deep_on_4x2.ringed(prefetching),
            deep_on_4x2.ringed(prefetching_one_stage),
            small_on_2x1.ringed(prefetching),
            small_on_2x2.ringed(prefetching),
        ];
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

        [lead]
            .into_iter()
            .chain(grids)
            .chain(layouts)
            .chain(boxes)
            .chain(small_cubes)
            .chain(rings)
            .chain(rows_in_order)
            .chain([split_lead])
            .chain(splits)
            .collect()
    }

    /// Items on matmul's own axes, which it places and stages in; a packed one, against a packed
    /// input whose scales group `K`, which it stages a block at a time.
    fn admits(item: &MatmulWorkItem) -> Result<(), Refusal> {
        match item.blocks {
            None => MatmulCubeSpace::admits(item),
            Some(_) => Self::block(item).map(|_| ()).ok_or(Self::UNGROUPED_K),
        }
    }

    /// Where the settings put the planes on warpgroups, that the warpgroup MMA's rules do not hold
    /// for this device or product; [`DECLINE`](Self::DECLINE) otherwise.
    fn decline(settings: &Self::Settings) -> &'static str {
        match settings.plane.holder {
            ComputeScope::WARPGROUP => {
                "this setting deals tiles to warpgroups, whose warpgroup MMA this device or \
                 product does not take"
            }
            _ => Self::DECLINE,
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
            None => Self::as_stored_candidates(hw, item, settings),
            Some(_) => Self::candidates_in_blocks(hw, item, settings),
        };
        candidates.into_iter().filter(move |_| !merges || atomic)
    }
    /// Its ring reads both operands out of their tiles a stage at a time, in stretches of the
    /// stage's extent along each one's contiguous axis, one element wide where the TMA engine
    /// fills it. Each plane drains its tile a fragment at a time, or a cell at a time from a manual
    /// fragment, into the output; merged along
    /// `K` in shared memory, it drains one element at a time, since an atomic takes one. Its
    /// planes' reads are the plane microkernel's.
    fn vectorize(
        &self,
        cube_space: &MatmulCubeSpace,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        self.stage(cube_space)
            .require_runs(self.plane.orders, vectorization);
        self.fill.vectorize(vectorization);
        self.store.vectorize(vectorization);
        match self.drain() {
            CubeStagedDrain::EachPlane => {
                vectorization.require(VectorRequirement::Access {
                    operand: MatmulOperand::Out,
                    elements: cube_space.leaf.extent(
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
            }
        }
        self.plane.vectorize(&(), vectorization);
    }

    /// Its ring, staging a step of `K` at a time for all its planes: rows of the lhs contiguous
    /// along the whole of `K` a cube walks, of the rhs along the stage's `n`, or along the whole of
    /// `K` for a col rhs bound in its buffer's order. Its drain: each plane writing its tile a
    /// row of it at a time, or every plane adding its whole partial box into the accumulator
    /// atomically before the cube copies it out a row at a time. Then its planes' microkernel, for
    /// the work.
    fn estimate(
        &self,
        item: &MatmulWorkItem,
        cube_space: &MatmulCubeSpace,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        let extents = &cube_space.space;
        let stage = &self.stage(cube_space);
        let (out_run, drain_atomics) = match self.drain() {
            CubeStagedDrain::EachPlane => (self.plane.tile_extent(N), 0),
            CubeStagedDrain::Merged => (
                extents.extent(N),
                (self.planes_along_k.get() * extents.extent(M) * extents.extent(N)) as u64,
            ),
        };
        let staging = MatmulStagingCount {
            drain_atomics,
            ..MatmulStagingCount::new(
                &cube_space.space,
                stage,
                self.plane.orders,
                &self.ring,
                out_run,
            )
        };
        staging.count(item, &cube_space.space, estimate);
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
    /// The levels inside one cube's box, which a routine places the cubes around: the plane's own,
    /// distributed over the cube's planes along its rows and columns and, where they split it, the
    /// contraction, walked through all of it a stage at a time. Every box and stage it gives is
    /// the cube's, whatever its fill turns out to set aside.
    ///
    /// A stage of scale blocks walks `KB`, its planes' steps inside each block
    /// (`PlaneFragmentMicrokernel::tiling_over`), and its planes split `KB`.
    ///
    /// An instruction several planes issue together, a warpgroup MMA's, distributes the box over
    /// groups of them instead: the layout counts groups.
    fn tiling(&self) -> Levels {
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

    /// Its planes' order: they read the rhs's tiles out of its stages.
    fn orders(&self) -> MatmulOperands<BindingOrder> {
        self.plane.orders
    }

    fn delivery(&self) -> Delivery {
        self.fill.delivery.bound()
    }

    fn fills_by_tma(&self) -> bool {
        self.fill.delivery == CubeStagedDelivery::Tma
    }

    /// As their words where its planes' instruction is block-scaled and takes a word of them as its
    /// scale register: the word's blocks are one step of it, NVFP4's four of sixteen under a step
    /// of 64. As fields otherwise, each read and narrowed as the planes load it, kept beside the
    /// values: MXFP8's word of four `ue8m0` scales covers four of its steps of 32.
    fn block_scales(&self) -> BlockScales {
        match self.plane.instruction.form() {
            Some(form) if self.plane.instruction.scale_word_is_a_step(form) => BlockScales::Words,
            Some(_) | None => BlockScales::Fields,
        }
    }

    /// Its walk's ([`CubeStagedWalk::decoded`]).
    fn decoded_stage(&self) -> Option<ElemType> {
        self.walk().decoded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{MatmulLayout, MatmulReading, MatmulRows};
    use cubecl::features::MmaConfig;
    use cubecl::ir::FloatKind;
    use cubek_std::devices;

    /// An `f32` product read rounded runs its planes on the tf32 matrix instruction a device
    /// offers, its inputs staged decoded from their words; read as stored, it runs no plane, as
    /// the device offers no instruction for `f32`.
    #[test]
    fn a_rounded_f32_product_runs_the_tf32_instruction() {
        let (f32, tf32) = (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::TF32),
        );
        let hw = devices::offering_mma(
            devices::h100(),
            [MmaConfig {
                a_type: tf32,
                b_type: tf32,
                cd_type: f32,
                m: 16,
                n: 8,
                k: 8,
            }],
        );
        let item = |reading| MatmulWorkItem {
            elements: MatmulOperands::all(f32),
            sum: f32,
            layouts: MatmulOperands::all(MatmulLayout::Row),
            blocks: None,
            rows: MatmulRows::Tiled,
            destination: MatmulDestination::Buffer,
            axes: MatmulAxes::MNK,
            reading,
        };
        let settings = CubeStagedMicrokernelSettings::DEFAULT;
        let rounded =
            CubeStagedMicrokernel::best(&hw, &item(MatmulReading::Rounded), &settings).unwrap();
        assert_eq!(rounded.decoded_stage(), Some(tf32));
        // The exact product has no instruction for `f32` here, so no plane runs it: the unit
        // routines take it.
        let exact = CubeStagedMicrokernel::best(&hw, &item(MatmulReading::AsStored), &settings);
        assert!(exact.is_err(), "{exact:?}");
    }
}
