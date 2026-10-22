use crate::stored::Operand;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::{Axis, Space};
use cubek_std::matmul_axes::{K, M, N};
use cubek_std::{Hardware, Instructions, TMA_ROW_BYTES, block_scaled_nvfp4, manual_mma};

/// What a launch of this kind declines whatever the routine: a weight a read of which straddles
/// two scale tiles, or an `n` or a `k` that is not a whole number of [steps](MatmulStep).
pub const MATMUL_STEPS_UNCLOSED: Refusal = Refusal::Declined {
    why: "the packed routines read a weight whose reads sit inside one scale tile, and whose n and k are whole steps: the least box every load of the activation, of the values and of the scales fills whole",
};

/// What a product of a packed activation declines where the weight's read along `K` splits a word
/// of it: the two are contracted a read of the weight at a time.
pub const MATMUL_PACKED_LHS_SPLIT: Refusal = Refusal::Declined {
    why: "a packed activation is contracted a read of the weight along k at a time, and this weight's read splits a word of the activation",
};

/// The step of a product against a packed weight: the smallest box over `M`, `N` and `K` that
/// every load it makes, and the instruction contracting it, fill whole. On each axis it is the
/// least common multiple of what they cover, stated as a product: the instruction's extent times
/// the fewest calls of it that every load fills, so a box made of whole steps loads whole: the
/// activation at the device's width, the weight a whole read of its values and a whole read of its
/// scales at a time.
///
/// Every routine of this kind cuts its boxes as whole steps, whether the weight is in rows or in
/// the tiles it is stored in: [`Operand::stored`] states what a read of a stored weight brings,
/// and the step is read off that.
///
/// NVFP4 in tile order on a device of 128-bit loads: an `f16` activation loaded eight of `k` at a
/// time, a word of four scales covering sixty-four of `k`, and its values two words, one block,
/// down `k` a read: a step of one column by sixty-four of `k`, which takes eight loads of
/// activation, one of scales and four of values. Under the block-scaled instruction, its stages
/// filled by the TMA engine, it is two of the instruction's `16 × 8 × 64` calls deep, `128 × 8`;
/// under an `8 × 8 × 8` instruction, `64 × 8`. A load stores the weight in the step its device
/// takes, so a stored tile is whole steps by construction.
///
/// A stage the TMA engine fills is one more load ([`filled_by_tma`](Self::filled_by_tma)): a box
/// moves rows of [`TMA_ROW_BYTES`] along the axis the weight's words run along, the row it copies
/// at full speed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulStep {
    activation: Space,
    scales: Space,
    values: Space,
    /// What one move of the fill brings: a TMA box's row, one value where the units fill.
    fill: Space,
    /// What one call of the instruction contracts; one value along every axis where nothing
    /// contracts but the loads.
    instruction: Space,
    /// Calls of the instruction one step makes along `M`, `N` and `K`: the fewest whose extent
    /// every load fills whole.
    calls: Space,
}

impl MatmulStep {
    /// The step of a product of a `lhs` activation against `rhs` on `hw`, contracted one value at
    /// a time: the weight's read and its scales' as its operand states them, and the activation
    /// one of the device's loads along `K` where the weight's words run along it, one value where
    /// they run across it and each value of `k` meets a whole line of columns.
    pub fn new(hw: &Hardware, rhs: &Operand, lhs: ElemType) -> Self {
        let activation = match rhs.lined() == K {
            true => hw.load_values(lhs),
            false => 1,
        };
        Self::covering(Self {
            activation: Space::new(&[(K, activation)]),
            scales: rhs.scale_read_space().unwrap_or_else(|| Space::new(&[])),
            values: rhs.read_space(),
            fill: Space::new(&[]),
            instruction: Space::new(&[]),
            calls: Space::new(&[]),
        })
    }

    /// This step under `instruction` too, the box over `M`, `N` and `K` one call of it
    /// contracts: a tensor-core instruction's `m × n × k`.
    pub fn under(self, instruction: Space) -> Self {
        Self::covering(Self {
            instruction,
            ..self
        })
    }

    /// This step where the TMA engine fills the stages of a weight read as `rhs`, its values
    /// `element`: a box moves rows of [`TMA_ROW_BYTES`] along the axis the words run along, so the
    /// step is that many bytes of values deep along it. NVFP4's `e2m1` rows are a hundred and
    /// twenty-eight values of `k`.
    pub fn filled_by_tma(self, rhs: &Operand, element: ElemType) -> Self {
        let row = (TMA_ROW_BYTES * 8 / element.size_bits().max(1)).max(1);
        Self::covering(Self {
            fill: Space::new(&[(rhs.lined(), row)]),
            ..self
        })
    }

    /// The step a load stores a weight in on `hw`: a product of an `lhs` activation against the
    /// weight read as `rhs`, its values `values`, under the largest instruction `hw` offers to
    /// contract it, its stages filled by the TMA engine where `hw` has one. The decode routines
    /// take no instruction, and their step is a whole number of times inside this one, so one
    /// stored tile serves both.
    ///
    /// NVFP4 values on a device that contracts them as stored take the block-scaled instruction:
    /// `128 × 8`. Under an `8 × 8 × 8` instruction an `f16` activation takes `64 × 8` of NVFP4
    /// and `8 × 8` of a float weight. `None` on a device that offers no instruction to contract
    /// it: there a tile would feed no fragment, and the weight stays as it was laid down.
    pub fn stored_on(
        hw: &Hardware,
        rhs: &Operand,
        values: ElemType,
        lhs: ElemType,
    ) -> Option<Self> {
        let f32 = ElemType::Float(FloatKind::F32);
        let shape = match values == ElemType::Float(FloatKind::E2M1) && block_scaled_nvfp4(hw) {
            true => manual_mma(hw).for_triple(values, values, f32).largest(),
            false => hw
                .properties
                .features
                .matmul
                .cmma
                .iter()
                .copied()
                .for_triple(lhs, lhs, f32)
                .largest(),
        }?;
        let step = Self::new(hw, rhs, lhs);
        let step = match hw.bulk_tensor_copy() {
            true => step.filled_by_tma(rhs, values),
            false => step,
        };
        Some(step.under(Space::new(&[
            (M, shape.m as usize),
            (N, shape.n as usize),
            (K, shape.k as usize),
        ])))
    }

    /// `step` with its calls the fewest along each axis whose extent every load fills whole.
    fn covering(step: Self) -> Self {
        let calls = [M, N, K].map(|axis| {
            let call = Self::reach(&step.instruction, axis);
            let calls = (1..)
                .find(|&calls| {
                    [&step.activation, &step.scales, &step.values, &step.fill]
                        .into_iter()
                        .all(|load| (call * calls).is_multiple_of(Self::reach(load, axis)))
                })
                .expect("some count of calls fills every load");
            (axis, calls)
        });
        Self {
            calls: Space::new(&calls),
            ..step
        }
    }

    /// How far `space` reaches along `axis`: one where it does not name it.
    fn reach(space: &Space, axis: Axis) -> usize {
        match space.contains(axis) {
            true => space.extent(axis),
            false => 1,
        }
    }

    /// The step's extent along `axis`: the instruction's times its calls.
    pub fn along(&self, axis: Axis) -> usize {
        Self::reach(&self.instruction, axis) * self.calls.extent(axis)
    }

    /// Calls of the instruction one step makes along `axis`.
    pub fn calls(&self, axis: Axis) -> usize {
        self.calls.extent(axis)
    }

    /// Whether an `n × k` weight read as `rhs` is a whole number of steps, each read of it inside
    /// one scale tile.
    ///
    /// # Errors
    ///
    /// [`MATMUL_STEPS_UNCLOSED`].
    pub fn check(&self, rhs: &Operand, n: usize, k: usize) -> Result<(), Refusal> {
        let lined = rhs.lined();
        let straddles = rhs
            .scale(lined)
            .is_some_and(|tile| !tile.is_multiple_of(rhs.read(lined)));
        let closes = n.is_multiple_of(self.along(N)) && k.is_multiple_of(self.along(K));
        match closes && !straddles {
            true => Ok(()),
            false => Err(MATMUL_STEPS_UNCLOSED),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile_order::{self, VALUES_AXES as VALUES};
    use cubecl::features::{MmaConfig, Tma};
    use cubecl::ir::{FloatKind, OpaqueType, UIntKind};
    use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// How many loads of the activation, of the scales and of the values, and how many calls of
    /// the instruction, one step takes.
    fn counts(step: &MatmulStep) -> [usize; 4] {
        [
            &step.activation,
            &step.scales,
            &step.values,
            &step.instruction,
        ]
        .map(|part| Space::new(&[M, N, K].map(|axis| (axis, step.along(axis)))).tiles(part))
    }

    /// NVFP4 in tile order as the writer stores it for a 128-bit load.
    fn tile_order() -> Operand {
        Operand::packed(K, 8, K, tile_order::BLOCK, 4)
            .stored(
                N,
                4,
                Some((&VALUES, tile_order::values_storage((128, 8), 4).tiles())),
                None,
            )
            .expect("tile order is read")
    }

    /// `value` packed along `axis` of a `[k, n]` weight under `f16` scales of 32 along it.
    fn in_rows(value: QuantValue, axis: Axis) -> Operand {
        let (block, store) = match axis {
            K => (vec![32, 1], QuantStore::PackedU32(1)),
            _ => (vec![1, 32], QuantStore::PackedU32(0)),
        };
        let scheme = QuantScheme::default()
            .with_value(value)
            .per_block(block, ScaleDtype::F16)
            .with_store(store);
        Operand::of_scheme(&scheme, axis, [K, N]).expect("one axis a block")
    }

    /// The `16 × 8 × 64` block-scaled instruction over NVFP4.
    fn block_scaled() -> Space {
        Space::new(&[(M, 16), (N, 8), (K, 64)])
    }

    /// NVFP4 in tile order on a device of 128-bit loads: a step of one word of scales, sixty-four
    /// of `k`, by one column: eight loads of activation, one of scales and four of values, each
    /// read inside one block.
    #[test]
    fn an_nvfp4_step_is_one_word_of_scales_deep() {
        let step = MatmulStep::new(&devices::m2_pro(), &tile_order(), F16);
        assert_eq!([M, N, K].map(|axis| step.along(axis)), [1, 1, 64]);
        assert_eq!(counts(&step), [8, 1, 4, 1]);
    }

    /// MXFP4's blocks of thirty-two make the step twice as deep as NVFP4's in rows: a word of
    /// scales covers four blocks either way.
    #[test]
    fn an_mxfp4_step_is_twice_as_deep() {
        let hw = devices::m2_pro();
        let nvfp4 = MatmulStep::new(&hw, &Operand::packed(K, 8, K, 16, 4), F16);
        let mxfp4 = MatmulStep::new(&hw, &Operand::packed(K, 8, K, 32, 4), F16);
        assert_eq!((nvfp4.along(K), mxfp4.along(K)), (64, 128));
        assert_eq!(counts(&mxfp4), [16, 1, 16, 1]);
    }

    /// Under an `8 × 8 × 8` instruction the NVFP4 step keeps its depth, eight calls down, and
    /// takes the instruction's rows and columns.
    #[test]
    fn an_instruction_joins_the_step_where_the_loads_leave_it() {
        let step = MatmulStep::new(&devices::m2_pro(), &tile_order(), F16).under(Space::new(&[
            (M, 8),
            (N, 8),
            (K, 8),
        ]));
        assert_eq!([M, N, K].map(|axis| step.along(axis)), [8, 8, 64]);
        assert_eq!(counts(&step), [8, 1, 4, 8]);
        assert_eq!([M, N, K].map(|axis| step.calls(axis)), [1, 1, 8]);
    }

    /// A stage the TMA engine fills moves rows of sixty-four bytes: a hundred and twenty-eight
    /// `e2m1` values of `k`, which deepens a step of one block, scales one to a word, eightfold,
    /// and one of a word of four blocks twofold.
    #[test]
    fn a_tma_row_deepens_a_step_to_its_row() {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        let hw = devices::m2_pro();
        let one_block = Operand::packed(K, 8, K, 16, 1);
        let shallow = MatmulStep::new(&hw, &one_block, F16);
        let filled = shallow.clone().filled_by_tma(&one_block, e2m1);
        assert_eq!((shallow.along(K), filled.along(K)), (16, 128));
        assert_eq!(filled.along(N), shallow.along(N));

        let deep = MatmulStep::new(&hw, &tile_order(), F16);
        let filled = deep.clone().filled_by_tma(&tile_order(), e2m1);
        assert_eq!((deep.along(K), filled.along(K)), (64, 128));
    }

    /// A weight in rows steps a word of scales along the axis its words run along, and one read
    /// across it: `q4` under `f16` scales of 32, two to a word, steps sixty-four of `k` or of `n`.
    #[test]
    fn a_weight_in_rows_steps_a_word_of_scales_along_its_words() {
        let hw = devices::m2_pro();
        let along_k = MatmulStep::new(&hw, &in_rows(QuantValue::Q4S, K), F16);
        assert_eq!((along_k.along(K), along_k.along(N)), (64, 1));
        let along_n = MatmulStep::new(&hw, &in_rows(QuantValue::Q4S, N), F16);
        assert_eq!((along_n.along(K), along_n.along(N)), (1, 64));
    }

    /// A weight is refused where `n` or `k` cuts a step short, and served where both are whole.
    #[test]
    fn a_weight_short_of_a_whole_step_is_refused() {
        let rhs = tile_order();
        let step = MatmulStep::new(&devices::m2_pro(), &rhs, F16).under(block_scaled());
        assert_eq!(step.check(&rhs, 4096, 4096), Ok(()));
        for (n, k) in [(4092, 4096), (4096, 4104)] {
            assert_eq!(
                step.check(&rhs, n, k),
                Err(MATMUL_STEPS_UNCLOSED),
                "{n} × {k}"
            );
        }
    }

    /// A weight stored in tile order is a whole number of steps on every load width a device
    /// offers, at each tile a load picks: the step under the block-scaled instruction with TMA
    /// rows, under an `8 × 8 × 8` instruction, and under none. A stored tile never cuts one.
    #[test]
    fn a_stored_tile_is_whole_steps() {
        for (k, n) in [(128, 8), (64, 8), (64, 1)] {
            for load_width in [32, 64, 128] {
                let hw = {
                    let mut hw = devices::m2_pro();
                    devices::topology(&mut hw).load_width = load_width;
                    hw
                };
                let words = hw.load_values(ElemType::UInt(UIntKind::U32));
                let rhs = Operand::packed(K, 8, K, tile_order::BLOCK, 4)
                    .stored(
                        N,
                        words,
                        Some((&VALUES, tile_order::values_storage((k, n), words).tiles())),
                        None,
                    )
                    .expect("tile order is read at every load");
                let step = MatmulStep::new(&hw, &rhs, F16);
                assert_eq!(
                    step.check(&rhs, n, k),
                    Ok(()),
                    "{k} × {n}, {load_width}-bit loads"
                );
            }
        }
    }

    /// The step of an NVFP4 product under the block-scaled instruction, its stages filled by the
    /// TMA engine, is `128 × 8` in tile order and in rows alike, on every load width: the tile a
    /// load stores in on a device with that instruction, one call's operand.
    #[test]
    fn the_block_scaled_step_is_two_calls_deep() {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        for load_width in [32, 64, 128] {
            let hw = {
                let mut hw = devices::m2_pro();
                devices::topology(&mut hw).load_width = load_width;
                hw
            };
            let words = hw.load_values(ElemType::UInt(UIntKind::U32));
            let in_rows = Operand::packed(K, 8, K, tile_order::BLOCK, 4);
            let stored = in_rows
                .stored(
                    N,
                    words,
                    Some((&VALUES, tile_order::values_storage((128, 8), words).tiles())),
                    None,
                )
                .expect("tile order is read at every load");
            for rhs in [in_rows, stored] {
                let step = MatmulStep::new(&hw, &rhs, e2m1)
                    .filled_by_tma(&rhs, e2m1)
                    .under(block_scaled());
                assert_eq!(
                    (step.along(K), step.along(N)),
                    (128, 8),
                    "{load_width}-bit loads, {rhs:?}"
                );
                assert_eq!([M, N, K].map(|axis| step.calls(axis)), [1, 1, 2]);
            }
        }
    }

    /// A load stores a weight in its step under the device's largest instruction. NVFP4 takes
    /// `128 × 8` under the block-scaled one with TMA rows and `64 × 8` under an `8 × 8 × 8` one; a
    /// float weight `8 × 8`. Under none, neither is stored in tiles.
    #[test]
    fn a_load_stores_in_the_step_under_the_largest_instruction() {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        let mma = |a_type, m, n, k| MmaConfig {
            a_type,
            b_type: a_type,
            cd_type: F32,
            m,
            n,
            k,
        };
        let none = devices::m2_pro();
        let simdgroup = devices::offering_cmma(devices::m2_pro(), [mma(F16, 8, 8, 8)]);
        let mut block_scaled = devices::offering_nvfp4(simdgroup.clone());
        let features = devices::features(&mut block_scaled);
        features.tma = Tma::Base.into();
        features.types.opaque.insert(OpaqueType::Barrier);
        let nvfp4 = Operand::packed(K, 8, K, tile_order::BLOCK, 4);
        let float = Operand::plain();
        let tile = |hw: &Hardware, rhs: &Operand, values| {
            MatmulStep::stored_on(hw, rhs, values, F16).map(|step| (step.along(K), step.along(N)))
        };
        assert_eq!(tile(&block_scaled, &nvfp4, e2m1), Some((128, 8)));
        assert_eq!(tile(&simdgroup, &nvfp4, e2m1), Some((64, 8)));
        assert_eq!(tile(&none, &nvfp4, e2m1), None);
        assert_eq!(tile(&simdgroup, &float, F16), Some((8, 8)));
        assert_eq!(tile(&none, &float, F16), None);
    }
}
