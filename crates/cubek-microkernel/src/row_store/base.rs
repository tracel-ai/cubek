use cubecl::features::Plane;
use cubecl::ir::ElemType;
use cubecl::quant::scheme::QuantScheme;
use cubecl::tile::kind::Field;
use cubecl::tile::launch::{MaskedExtent, Refusal, VectorRequirement};
use cubecl::tile::{Axis, Partitioning, Space};
use cubecl::zspace::metadata::Metadata;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, KB, KI, MB, MI};

/// The axis a producer's rows run along: the rows of the product's lhs it writes, which a packed
/// matrix binds as the position inside the one block of the product's rows.
pub const ROWS: Axis = MI;
/// The axis a producer's row spans: the contraction of the product its rows feed.
pub const WIDTH: Axis = K;

/// What a packed store declines before preparing anything: an output whose scales are not one
/// level of blocks along the row, or a row that is not whole words of block scales, which every
/// read of them takes.
pub const PACKED_STORE_UNSERVED: Refusal = Refusal::Declined {
    why: "a packed store writes one level of block scales along each row, whole words of them a row",
};

/// What a routine declines a packed store with on a device without plane instructions: the
/// encoding calibrates each word of block scales across the lanes holding its blocks.
pub const PACKED_STORE_NEEDS_PLANES: Refusal = Refusal::Declined {
    why: "a packed store calibrates its block scales across a plane's lanes, and this device has no plane instructions",
};

/// What a routine whose owners meet in no cube declines a packed store with: the cube is where a
/// stage of its rows is encoded.
pub const PACKED_STORE_NEEDS_A_CUBE: Refusal = Refusal::Declined {
    why: "a packed store encodes a cube's stage once its owners have written it, and this routine \
          walks the launch's planes, which no cube holds",
};

/// How a producer stores the rows it computes: as its element, or packed under one level of block
/// scales of `block` elements along each row, `scales_per_word` of them a word, as the lhs of a
/// product reads them.
///
/// It decides the space the producer runs over: a plain store's rows and row span are [`ROWS`]
/// and [`WIDTH`]; a packed one's are the axes a packed matrix binds over, [`M`] and [`K`] each
/// split into its blocks and the position inside one, so its output binds as a product's packed
/// lhs does and the producer's own operands are restated over the same space.
///
/// [`M`]: cubek_std::matmul_axes::M
/// [`K`]: cubek_std::matmul_axes::K
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowStore {
    /// Each line written as computed.
    Plain,
    /// Each row encoded in blocks of `block` elements, under the scales the store calibrates,
    /// stored `scales_per_word` to a word: a run an owner writes is whole words of them, since a
    /// word of scales is read and written whole.
    Packed {
        block: usize,
        scales_per_word: usize,
    },
}

/// What one owner of a cube holds in shared memory under a packed store: the elements of its row
/// it stages, and the bytes it holds of its own beside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerMemory {
    /// Elements of its row the owner stages.
    pub staged: usize,
    /// The element the stage holds.
    pub element: ElemType,
    /// Shared memory the owner holds of its own, past its stage.
    pub scratch_bytes: usize,
}

impl OwnerMemory {
    /// Bytes the owner holds: its stage and its own.
    pub fn bytes(&self) -> usize {
        self.staged * self.element.size() + self.scratch_bytes
    }
}

impl RowStore {
    /// The store a packed output under `scheme` takes for rows of `width` elements: one level of
    /// block scales along the last axis, whole words of them a row.
    ///
    /// # Errors
    ///
    /// [`PACKED_STORE_UNSERVED`].
    pub fn packed(scheme: &QuantScheme, width: usize) -> Result<Self, Refusal> {
        let size = scheme
            .block_size()
            .filter(|_| scheme.num_levels() == 1 && scheme.packing_dim() == Some(0))
            .ok_or(PACKED_STORE_UNSERVED)?;
        let block = match size.as_slice() {
            [leading @ .., block] if leading.iter().all(|&extent| extent == 1) => *block as usize,
            _ => return Err(PACKED_STORE_UNSERVED),
        };
        let scale = scheme.block_scale().ok_or(PACKED_STORE_UNSERVED)?;
        let store = Self::Packed {
            block,
            scales_per_word: Field::of_scale(scale.dtype).per_word(),
        };
        match block > 0 && store.holds_cube_run(width) {
            true => Ok(store),
            false => Err(PACKED_STORE_UNSERVED),
        }
    }

    /// The axes `rows` rows are stated over: [`ROWS`], or the one block of the product's rows and
    /// the rows inside it.
    pub fn rows_axes(&self, rows: usize) -> Vec<(Axis, usize)> {
        match self {
            Self::Plain => vec![(ROWS, rows)],
            Self::Packed { .. } => vec![(MB, 1), (ROWS, rows)],
        }
    }

    /// The axes `extent` elements of a row are stated over: [`WIDTH`], or its blocks and the
    /// position inside one.
    ///
    /// # Panics
    ///
    /// Where `extent` is not whole blocks: a store's every run is.
    pub fn width_axes(&self, extent: usize) -> Vec<(Axis, usize)> {
        match self {
            Self::Plain => vec![(WIDTH, extent)],
            Self::Packed { block, .. } => {
                assert!(
                    extent.is_multiple_of(*block),
                    "a packed store states whole blocks of a row, not {extent} elements"
                );
                vec![(KB, extent / block), (KI, *block)]
            }
        }
    }

    /// The axis a row is cut along between owners or cubes: [`WIDTH`], or its blocks.
    pub fn cut(&self) -> Axis {
        match self {
            Self::Plain => WIDTH,
            Self::Packed { .. } => KB,
        }
    }

    /// Whether a run of `elements` of a row is one an owner may write into a cube's stage: any,
    /// or whole blocks, which the stage states the row in.
    pub fn holds_owner_run(&self, elements: usize) -> bool {
        match self {
            Self::Plain => true,
            Self::Packed { block, .. } => elements.is_multiple_of(*block),
        }
    }

    /// Whether a run of `elements` of a row is one a cube may encode: any, or whole words of
    /// block scales, since a word of scales is bound, read and written whole and no two cubes
    /// write one.
    pub fn holds_cube_run(&self, elements: usize) -> bool {
        match self {
            Self::Plain => true,
            Self::Packed {
                block,
                scales_per_word,
            } => elements.is_multiple_of(block * scales_per_word),
        }
    }

    /// Whether `hw` runs this store: every device a plain one, and a packed one where the device
    /// has plane instructions, which its encoding calibrates a word of scales with, the lanes
    /// holding its blocks meeting through the plane.
    ///
    /// # Errors
    ///
    /// [`PACKED_STORE_NEEDS_PLANES`].
    pub fn runs_on(&self, hw: &Hardware) -> Result<(), Refusal> {
        let planes = hw.properties.features.plane.contains(Plane::Ops);
        match (self, planes) {
            (Self::Packed { .. }, false) => Err(PACKED_STORE_NEEDS_PLANES),
            _ => Ok(()),
        }
    }

    /// The owners a cube holds at most on `hw` under this store, each holding `owner` in shared
    /// memory: as many as the device's budget holds, or no bound for a plain store, which stages
    /// nothing.
    pub fn owners_staged(&self, hw: &Hardware, owner: OwnerMemory) -> usize {
        match self {
            Self::Plain => usize::MAX,
            Self::Packed { .. } => (hw.shared_memory_budget() / owner.bytes().max(1)).max(1),
        }
    }

    /// The axes a kernel leaves to the launch: the rows of a plain store. A packed store's scales
    /// vary along the rows, and the tile engine windows them through extents it knows at
    /// expansion, so its rows are compiled in, as a product's packed lhs's are.
    pub fn dynamic_axes(&self) -> &'static [Axis] {
        match self {
            Self::Plain => &[ROWS],
            Self::Packed { .. } => &[],
        }
    }

    /// The mask `partitioning`'s tiles put on `operand`'s row where they overhang it anywhere
    /// along the axes it is stated over, or none where they do not: a plain row's lines are tested
    /// against its width, which a dividing width never straddles; a packed row's edge spans its
    /// blocks and the position inside one, two axes no line can be tested along whole, so it binds
    /// one element at a time.
    pub fn mask<O: Copy + Eq>(
        &self,
        operand: O,
        partitioning: &Partitioning,
    ) -> Option<VectorRequirement<O>> {
        match self {
            Self::Plain => partitioning
                .overhangs(WIDTH)
                .then(|| VectorRequirement::masked(operand, partitioning.space(), WIDTH)),
            Self::Packed { .. } => (partitioning.overhangs(KB) || partitioning.overhangs(KI))
                .then_some(VectorRequirement::Masked {
                    operand,
                    extent: MaskedExtent::Unknown,
                }),
        }
    }

    /// The stage a cube writes its rows into first: none for a plain store, which writes them
    /// straight out, or the cube's box over the axes a packed matrix binds over — the blocks of
    /// the rows and the rows inside them, the blocks of the row and the position inside one —
    /// each at the cube's extent along it. An axis of the cube's box the packed output does not
    /// run along, a batch entry the partitioning walks apart or the halves of a gated row, is
    /// left out, as the encoding's window, which the stage must be, leaves it out.
    pub fn cube_stage(&self, cube_box: &Space) -> CubeStage {
        match self {
            Self::Plain => CubeStage::Direct,
            Self::Packed { .. } => {
                let axes: Vec<(Axis, usize)> = [MB, MI, KB, KI]
                    .into_iter()
                    .map(|axis| match cube_box.contains(axis) {
                        true => (axis, cube_box.extent(axis)),
                        false => (axis, 1),
                    })
                    .collect();
                CubeStage::Encoded {
                    cube_box: Space::new(&axes),
                }
            }
        }
    }

    /// Bytes of shared memory a cube's stage of `elements` elements of `element` takes: none for
    /// a plain store.
    pub fn stage_bytes(&self, elements: usize, element: ElemType) -> usize {
        match self {
            Self::Plain => 0,
            Self::Packed { .. } => elements * element.size(),
        }
    }

    /// `meta`, dense rows of `width` elements whose leading dims `leading` names, as the store
    /// binds them: as they lie over `leading` then [`WIDTH`], or with the last dim split into the
    /// blocks and the position inside one.
    pub fn restated(&self, meta: &Metadata, leading: &[Axis]) -> (Metadata, Vec<Axis>) {
        match self {
            Self::Plain => (meta.clone(), [leading, &[WIDTH]].concat()),
            Self::Packed { block, .. } => {
                let shape = meta.shape();
                let strides = meta.strides();
                let rank = shape.len();
                let width = shape[rank - 1];
                let mut dims: Vec<usize> = shape[..rank - 1].to_vec();
                let mut steps: Vec<usize> = strides[..rank - 1].to_vec();
                dims.extend([width / block, *block]);
                steps.extend([block * strides[rank - 1], strides[rank - 1]]);
                (Metadata::new(dims, steps), [leading, &[KB, KI]].concat())
            }
        }
    }
}

/// The stage a cube's owners write their rows into before they reach the output: none, or one of
/// the cube's box, encoded into the packed output once every owner has written it. A comptime
/// decision of the kernel, read off the routine's [`RowStore`] and the cube's box.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CubeStage {
    /// Each owner writes its lines straight into its window of the output.
    Direct,
    /// Each owner writes its lines into its run of a stage over `cube_box`, laid out in rows.
    Encoded { cube_box: Space },
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::quant::scheme::{QuantStore, QuantValue, ScaleDtype};
    use cubecl::tile::Levels;

    /// A plain row its tiles overhang is masked at its width, which a width dividing it never
    /// straddles; a packed row its blocks overhang at an unknown edge, one element at a time; a
    /// row no tile overhangs is not masked at all.
    #[test]
    fn a_row_its_tiles_overhang_is_masked_at_its_edge() {
        let plain = |width: usize| {
            let space = Space::new(&[(ROWS, 4), (WIDTH, width)]);
            let levels = Levels::leaf(&[(ROWS, 1), (WIDTH, 64)])
                .planes(&[(ROWS, 1)])
                .cubes(&[WIDTH, ROWS])
                .build();
            RowStore::Plain.mask('x', &Partitioning::new(space, levels))
        };
        assert_eq!(
            plain(100),
            Some(VectorRequirement::Masked {
                operand: 'x',
                extent: MaskedExtent::Known(100),
            })
        );
        assert_eq!(plain(128), None);
        let packed = RowStore::Packed {
            block: 32,
            scales_per_word: 4,
        };
        let space = Space::new(&[(MB, 1), (MI, 4), (KB, 3), (KI, 32)]);
        let levels = Levels::leaf(&[(MI, 1), (KB, 2), (KI, 32)])
            .planes(&[(MI, 1)])
            .cubes(&[KB, MI])
            .build();
        assert_eq!(
            packed.mask('x', &Partitioning::new(space, levels)),
            Some(VectorRequirement::Masked {
                operand: 'x',
                extent: MaskedExtent::Unknown,
            })
        );
    }

    /// A one-level scheme along the row is a packed store, its scales as many a word as their
    /// width fits; two levels, a block across the rows, or a row of a part-word of scales is not.
    #[test]
    fn a_packed_store_is_one_level_of_whole_words_of_scales() {
        let mxfp8 = scheme(QuantValue::E4M3, 32, ScaleDtype::UE8M0);
        assert_eq!(
            RowStore::packed(&mxfp8, 2560),
            Ok(RowStore::Packed {
                block: 32,
                scales_per_word: 4
            })
        );
        let q8 = scheme(QuantValue::Q8S, 32, ScaleDtype::F32);
        assert_eq!(
            RowStore::packed(&q8, 96),
            Ok(RowStore::Packed {
                block: 32,
                scales_per_word: 1
            })
        );
        // Three blocks of 8-bit scales are part of a word.
        assert_eq!(RowStore::packed(&mxfp8, 96), Err(PACKED_STORE_UNSERVED));
        let nvfp4 = scheme(QuantValue::E2M1, 16, ScaleDtype::UE4M3).per_tensor(ScaleDtype::F32);
        assert_eq!(RowStore::packed(&nvfp4, 2560), Err(PACKED_STORE_UNSERVED));
        let across = QuantScheme::default()
            .with_value(QuantValue::E4M3)
            .per_block([32, 32], ScaleDtype::UE8M0)
            .with_store(QuantStore::PackedU32(0));
        assert_eq!(RowStore::packed(&across, 2560), Err(PACKED_STORE_UNSERVED));
    }

    /// A packed store states its rows over the axes a packed lhs binds over, the row split into
    /// its blocks, and compiles its rows in; a plain one states them as they are and leaves them
    /// to the launch.
    #[test]
    fn a_store_states_the_rows_it_writes() {
        let packed = RowStore::Packed {
            block: 32,
            scales_per_word: 4,
        };
        assert_eq!(packed.rows_axes(3), vec![(MB, 1), (MI, 3)]);
        assert_eq!(packed.width_axes(256), vec![(KB, 8), (KI, 32)]);
        assert_eq!(packed.cut(), KB);
        assert!(packed.dynamic_axes().is_empty());
        assert!(packed.holds_cube_run(128) && !packed.holds_cube_run(96));
        assert!(packed.holds_owner_run(96) && !packed.holds_owner_run(48));
        assert_eq!(RowStore::Plain.width_axes(256), vec![(K, 256)]);
        assert_eq!(RowStore::Plain.dynamic_axes(), &[ROWS]);
        assert!(RowStore::Plain.holds_cube_run(3) && RowStore::Plain.holds_owner_run(3));
    }

    fn scheme(value: QuantValue, block: u8, scale: ScaleDtype) -> QuantScheme {
        QuantScheme::default()
            .with_value(value)
            .per_block([block], scale)
            .with_store(QuantStore::PackedU32(0))
    }
}
