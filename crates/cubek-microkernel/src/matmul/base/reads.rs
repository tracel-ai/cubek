use cubecl::tile::launch::Decline;

/// What a launch of this kind declines whatever the routine: a packed input a read of which
/// straddles two scale tiles, or an `n` or a `k` that the [micro region](cubecl::tile::MicroRegion)
/// does not divide.
pub const MATMUL_READS_UNCLOSED: Decline = Decline::Unserved {
    why: "the packed routines read packed inputs whose reads sit inside one scale tile, and whose n and k are whole reads: the least box every load of the inputs' values and of their scales fills whole",
};

/// What a product of two packed inputs declines where a read of one along `K` splits a word of
/// the other: the two are contracted a read at a time.
pub const MATMUL_PACKED_READS_SPLIT: Decline = Decline::Unserved {
    why: "two packed inputs are contracted a read along k at a time, and a read of one splits a word of the other",
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{MatmulOperand, MatmulOperands};
    use cubecl::features::{MmaConfig, Tma};
    use cubecl::ir::ElemType;
    use cubecl::ir::{FloatKind, OpaqueType, UIntKind};
    use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use cubecl::tile::{Dimension, MacroRegion, MicroRegion, Space};
    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::layout::stored::Operand;
    use cubek_std::layout::tile_order;
    use cubek_std::matmul_dimensions::{K, M, N};

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A float `[m, k]` input of `element` against a `[k, n]` input read as `packed`.
    fn against(packed: &Operand) -> MatmulOperands<Operand> {
        MatmulOperands {
            rhs: *packed,
            ..MatmulOperands::all(Operand::plain())
        }
    }

    /// The micro region of a float `[m, k]` input of `element` against a `[k, n]` input read as
    /// `packed`.
    fn micro_against(hw: &Hardware, packed: &Operand, element: ElemType) -> MicroRegion {
        against(packed).micro_region(hw, &MatmulOperands::all(element))
    }

    /// How many reads a region takes of a float input of `element`, and of `packed`'s values and
    /// its scales.
    fn reads(
        hw: &Hardware,
        region: impl Fn(Dimension) -> usize,
        packed: &Operand,
        element: ElemType,
    ) -> [usize; 3] {
        let region = Space::new(&[M, N, K].map(|dimension| (dimension, region(dimension))));
        let float = Space::new(&[(packed.lined(), hw.load_values(element))]);
        let scales = packed
            .scale_read_space()
            .expect("a packed input has scales");
        [float, packed.read_space(), scales].map(|tile| region.tiles(&tile))
    }

    /// NVFP4 in tile order as the writer stores it for a 128-bit load.
    fn tile_order() -> Operand {
        Operand::packed(K, 8, K, cubek_std::NVFP4_BLOCK, 4)
            .in_tile_order(
                tile_order::TileOrder::laid(
                    &tile_order::nvfp4_scheme(),
                    tile_order::TileWords::RowByRow,
                    (128, 8),
                )
                .expect("NVFP4 is tile order"),
                N,
                4,
            )
            .expect("tile order is read")
    }

    /// `value` packed along `dimension` of a `[k, n]` rhs under `f16` scales of 32 along it.
    fn in_rows(value: QuantValue, dimension: Dimension) -> Operand {
        let (block, store) = match dimension {
            K => (vec![32, 1], QuantStore::PackedU32(1)),
            _ => (vec![1, 32], QuantStore::PackedU32(0)),
        };
        let scheme = QuantScheme::default()
            .with_value(value)
            .per_block(block, ScaleDtype::F16)
            .with_store(store);
        Operand::of_scheme(&scheme, dimension, [K, N]).expect("one dimension a block")
    }

    /// The `16 × 8 × 64` block-scaled instruction over NVFP4.
    fn block_scaled() -> Space {
        Space::new(&[(M, 16), (N, 8), (K, 64)])
    }

    /// NVFP4 in tile order on a device of 128-bit loads: a step of one word of scales, sixty-four
    /// of `k`, by one column: eight loads of the `f16` input, four of values and one of scales,
    /// each read inside one block.
    #[test]
    fn an_nvfp4_step_is_one_word_of_scales_deep() {
        let hw = devices::m2_pro();
        let micro = micro_against(&hw, &tile_order(), F16);
        assert_eq!(
            [M, N, K].map(|dimension| micro.along(dimension)),
            [1, 1, 64]
        );
        let along = |dimension| micro.along(dimension);
        assert_eq!(reads(&hw, along, &tile_order(), F16), [8, 4, 1]);
    }

    /// MXFP4's blocks of thirty-two make the step twice as deep as NVFP4's in rows: a word of
    /// scales covers four blocks either way.
    #[test]
    fn an_mxfp4_step_is_twice_as_deep() {
        let hw = devices::m2_pro();
        let nvfp4 = micro_against(&hw, &Operand::packed(K, 8, K, 16, 4), F16);
        let mxfp4 = micro_against(&hw, &Operand::packed(K, 8, K, 32, 4), F16);
        assert_eq!((nvfp4.along(K), mxfp4.along(K)), (64, 128));
        let along = |dimension| mxfp4.along(dimension);
        let packed = Operand::packed(K, 8, K, 32, 4);
        assert_eq!(reads(&hw, along, &packed, F16), [16, 16, 1]);
    }

    /// Under an `8 × 8 × 8` instruction the NVFP4 step keeps its depth, eight calls down, and
    /// takes the instruction's rows and columns.
    #[test]
    fn an_instruction_joins_the_region_where_the_reads_leave_it() {
        let region = MacroRegion::new(
            &micro_against(&devices::m2_pro(), &tile_order(), F16),
            Space::new(&[(M, 8), (N, 8), (K, 8)]),
        );
        assert_eq!(
            [M, N, K].map(|dimension| region.along(dimension)),
            [8, 8, 64]
        );
        assert_eq!(
            [M, N, K].map(|dimension| region.instructions(dimension)),
            [1, 1, 8]
        );
    }

    /// A stage the TMA engine fills moves rows of sixty-four bytes: a hundred and twenty-eight
    /// `e2m1` values of `k`, which deepens a step of one block, scales one to a word, eightfold,
    /// and one of a word of four blocks twofold.
    #[test]
    fn a_tma_row_deepens_a_step_to_its_row() {
        let e2m1 = ElemType::Float(FloatKind::E2M1);
        let hw = devices::m2_pro();
        let one_block = Operand::packed(K, 8, K, 16, 1);
        let value = Space::new(&[(K, 1)]);
        let shallow = MacroRegion::new(&micro_against(&hw, &one_block, F16), value.clone());
        let filled = shallow.divided_by(one_block.tma_row(e2m1));
        assert_eq!((shallow.along(K), filled.along(K)), (16, 128));
        assert_eq!(filled.along(N), shallow.along(N));

        let deep = MacroRegion::new(&micro_against(&hw, &tile_order(), F16), value);
        let filled = deep.divided_by(tile_order().tma_row(e2m1));
        assert_eq!((deep.along(K), filled.along(K)), (64, 128));
    }

    /// Which input is packed never changes the step along `K`: NVFP4 down the `k` of an `[m, k]`
    /// input against a float `[k, n]` one steps as deep as NVFP4 down the `k` of a `[k, n]` input
    /// against a float `[m, k]` one, and two NVFP4 inputs step alike.
    #[test]
    fn which_input_is_packed_never_changes_the_depth() {
        let hw = devices::m2_pro();
        let nvfp4 = Operand::packed(K, 8, K, 16, 4);
        let float = MatmulOperands::all(F16);
        let packed_rhs = against(&nvfp4).micro_region(&hw, &float);
        let packed_lhs = MatmulOperands {
            lhs: nvfp4,
            ..MatmulOperands::all(Operand::plain())
        }
        .micro_region(&hw, &float);
        let both = MatmulOperands {
            lhs: nvfp4,
            ..against(&nvfp4)
        }
        .micro_region(&hw, &float);
        assert_eq!(packed_lhs.along(K), packed_rhs.along(K));
        assert_eq!(both.along(K), packed_rhs.along(K));
    }

    /// Two packed inputs are refused where a read of one along `K` splits a word of the other.
    #[test]
    fn two_packed_reads_that_split_a_word_are_refused() {
        let hw = devices::m2_pro();
        let (word, half) = (
            Operand::packed(K, 8, K, 16, 4),
            Operand::packed(K, 4, K, 16, 4),
        );
        let both = MatmulOperands {
            lhs: half,
            ..against(&word)
        };
        let float = MatmulOperands::all(F16);
        assert_eq!(both.check_reads(&hw, &float, 64, 64), Ok(()));
        let split = MatmulOperands {
            lhs: Operand::packed(K, 3, K, 24, 4),
            ..against(&word)
        };
        assert_eq!(
            split.check_reads(&hw, &float, 64, 1536),
            Err(MATMUL_PACKED_READS_SPLIT)
        );
    }

    /// An rhs in rows steps a word of scales along the dimension its words run along, and one read
    /// across it: `q4` under `f16` scales of 32, two to a word, steps sixty-four of `k` or of `n`.
    #[test]
    fn an_rhs_in_rows_steps_a_word_of_scales_along_its_words() {
        let hw = devices::m2_pro();
        let along_k = micro_against(&hw, &in_rows(QuantValue::Q4S, K), F16);
        assert_eq!((along_k.along(K), along_k.along(N)), (64, 1));
        let along_n = micro_against(&hw, &in_rows(QuantValue::Q4S, N), F16);
        assert_eq!((along_n.along(K), along_n.along(N)), (1, 64));
    }

    /// An rhs is served where the micro region divides `n` and `k`, though the block-scaled
    /// instruction's region overhangs `n`.
    #[test]
    fn an_rhs_whole_reads_wide_is_served() {
        let float = MatmulOperands::all(F16);
        let rhs = against(&tile_order());
        assert_eq!(
            rhs.check_reads(&devices::m2_pro(), &float, 4092, 4096),
            Ok(())
        );
    }

    /// An rhs is refused where `k` cuts a read short.
    #[test]
    fn an_rhs_short_of_a_whole_read_is_refused() {
        let float = MatmulOperands::all(F16);
        let rhs = against(&tile_order());
        assert_eq!(
            rhs.check_reads(&devices::m2_pro(), &float, 4096, 4104),
            Err(MATMUL_READS_UNCLOSED)
        );
    }

    /// An rhs stored in tile order is a whole number of steps on every load width a device
    /// offers, at each tile a load picks: the step under the block-scaled instruction with TMA
    /// rows, under an `8 × 8 × 8` instruction, and under none. A stored tile never cuts one.
    #[test]
    fn a_stored_tile_is_whole_reads() {
        for (k, n) in [(128, 8), (64, 8), (64, 1)] {
            for load_width in [32, 64, 128] {
                let hw = {
                    let mut hw = devices::m2_pro();
                    devices::topology(&mut hw).load_width = load_width;
                    hw
                };
                let words = hw.load_values(ElemType::UInt(UIntKind::U32));
                let rhs = Operand::packed(K, 8, K, cubek_std::NVFP4_BLOCK, 4)
                    .in_tile_order(
                        tile_order::TileOrder::laid(
                            &tile_order::nvfp4_scheme(),
                            tile_order::TileWords::RowByRow,
                            (k, n),
                        )
                        .expect("NVFP4 is tile order"),
                        N,
                        words,
                    )
                    .expect("tile order is read at every load");
                assert_eq!(
                    against(&rhs).check_reads(&hw, &MatmulOperands::all(F16), n, k),
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
            let in_rows = Operand::packed(K, 8, K, cubek_std::NVFP4_BLOCK, 4);
            let stored = in_rows
                .in_tile_order(
                    tile_order::TileOrder::laid(
                        &tile_order::nvfp4_scheme(),
                        tile_order::TileWords::RowByRow,
                        (128, 8),
                    )
                    .expect("NVFP4 is tile order"),
                    N,
                    words,
                )
                .expect("tile order is read at every load");
            for rhs in [in_rows, stored] {
                let region = MacroRegion::new(&micro_against(&hw, &rhs, e2m1), block_scaled())
                    .divided_by(rhs.tma_row(e2m1));
                assert_eq!(
                    (region.along(K), region.along(N)),
                    (128, 8),
                    "{load_width}-bit loads, {rhs:?}"
                );
                assert_eq!(
                    [M, N, K].map(|dimension| region.instructions(dimension)),
                    [1, 1, 2]
                );
            }
        }
    }

    /// A load stores an rhs in its step under the device's largest instruction. NVFP4 takes
    /// `128 × 8` under the block-scaled one with TMA rows and `64 × 8` under an `8 × 8 × 8` one; a
    /// float rhs `8 × 8`. Under none, neither is stored in tiles.
    #[test]
    fn a_load_stores_in_the_region_under_the_largest_instruction() {
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
        let nvfp4 = Operand::packed(K, 8, K, cubek_std::NVFP4_BLOCK, 4);
        let float = Operand::plain();
        let tile = |hw: &Hardware, rhs: &Operand, values| {
            MatmulOperand::Rhs
                .stored_region(hw, *rhs, values, F16)
                .map(|region| (region.along(K), region.along(N)))
        };
        assert_eq!(tile(&block_scaled, &nvfp4, e2m1), Some((128, 8)));
        assert_eq!(tile(&simdgroup, &nvfp4, e2m1), Some((64, 8)));
        assert_eq!(tile(&none, &nvfp4, e2m1), None);
        assert_eq!(tile(&simdgroup, &float, F16), Some((8, 8)));
        assert_eq!(tile(&none, &float, F16), None);
    }
}
