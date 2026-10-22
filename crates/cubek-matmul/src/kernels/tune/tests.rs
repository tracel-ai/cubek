//! Tests for the matmul tune candidates: how each is keyed, prioritised and estimated.

mod candidate {
    use super::super::*;
    use crate::kernels::select::problem::OperandKind;
    use crate::kernels::tune::key::{MatmulBatching, MatmulTuneKey, OperandKey};
    use crate::kernels::{
        CubeStagedMatmulRoutine, MatmulRoutineProblem, UnitRegistersMatmulRoutine,
    };
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::MatmulMicrokernel;
    use cubek_microkernel::matmul::MatmulOrder;
    use cubek_std::Hardware;
    use cubek_std::autotune::{CandidatePriority, TuneCandidate};
    use cubek_std::devices;
    use cubek_std::matmul_axes::{K, N};
    use cubek_std::routine::Routine;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the `16×16×16` `f16` instruction summed in `f32`.
    fn offering(hw: Hardware) -> Hardware {
        devices::offering_cmma(
            hw,
            [MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: F32,
                m: 16,
                n: 16,
                k: 16,
            }],
        )
    }

    /// A key over an `f16` `m × k @ k × n` product of `batch` batches, row-major.
    fn key(batch: usize, m: usize, n: usize, k: usize) -> MatmulTuneKey {
        MatmulTuneKey::new(
            F16,
            k,
            n,
            m,
            batch,
            OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
            OperandKey::float(Some(N), MatmulBatching::new(&[batch])),
        )
    }

    #[test]
    fn a_candidate_per_setting_the_default_first() {
        let hw = offering(devices::strix_halo());
        let listed: Vec<_> = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw)
            .iter()
            .map(|candidate| *candidate.strategy())
            .collect();
        assert_eq!(listed, CubeStagedMatmulRoutine.settings(&hw));
    }

    /// A setting the device declines is tried after every candidate that runs, and claims no
    /// launch an earlier candidate could duplicate: the cube routine's planes splitting `K`, on a
    /// device whose shared memory takes no atomic add.
    #[test]
    fn a_declined_setting_is_tried_last_and_claims_no_launch() {
        let mut hw = offering(devices::strix_halo());
        devices::features(&mut hw).types.shared_atomic.clear();
        let key = key(1, 2048, 2048, 2048);
        let split = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw)
            .into_iter()
            .filter(|candidate| candidate.strategy().microkernel.planes_along_k.get() > 1);
        for candidate in split {
            let settings = candidate.strategy();
            assert_eq!(
                candidate.priority(&key),
                CandidatePriority::LAST,
                "{settings}"
            );
            assert_eq!(candidate.blueprint(&key), None, "{settings}");
        }
    }

    /// The priority is the table's one rule over the cube's output block: first where the block
    /// tiles the product, last where one cube covers all of it.
    #[test]
    fn the_priority_reads_the_block_one_cube_covers() {
        let hw = offering(devices::strix_halo());
        let lead = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
        assert_eq!(
            lead.priority(&key(1, 2048, 2048, 2048)),
            CandidatePriority::FIRST
        );
        assert_eq!(
            lead.priority(&key(1, 16, 16, 2048)),
            CandidatePriority::LAST
        );
    }

    /// A col weight is read where it lies, so the key that names one prepares the launch its
    /// tensors will: the cube routine's cmma fragment and the unit routine's register block both
    /// bind it in its buffer's order, in vectors along `K`.
    #[test]
    fn a_col_weight_key_prepares_the_launch_its_tensors_will() {
        let col = MatmulTuneKey::new(
            F16,
            256,
            256,
            256,
            1,
            OperandKey::float(Some(K), MatmulBatching::Shared),
            OperandKey::float(Some(K), MatmulBatching::Shared),
        );
        let hw = offering(devices::strix_halo());
        let cube = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
        let (in_buffer_order, _) = cube.blueprint(&col).expect("runs here");
        assert_eq!(
            in_buffer_order.microkernel.orders().rhs,
            MatmulOrder::InBufferOrder
        );
        assert!(in_buffer_order.widths.rhs > 1);

        let unit = MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw).swap_remove(0);
        let (merged, _) = unit.blueprint(&col).expect("runs here");
        assert_eq!(merged.microkernel.orders().rhs, MatmulOrder::InBufferOrder);
        assert!(merged.widths.rhs > 1);
    }

    /// On devices a dev machine cannot show: the unit routine, this kind's floor, prepares
    /// everywhere; a routine that declines a device is tried last and claims no launch, and one that
    /// prepares is placed by its block. A CPU holds fewer units a cube than one cube routine's cube
    /// of planes, so there the cube routine declines.
    #[test]
    fn every_device_keeps_the_floor_and_a_decline_claims_nothing() {
        let key = key(1, 2048, 2048, 2048);
        for hw in [
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for unit in MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, &hw) {
                assert!(unit.blueprint(&key).is_some(), "{}", unit.strategy());
            }
            let placed = |candidate_priority: CandidatePriority, claims: bool| match claims {
                true => assert_eq!(candidate_priority, CandidatePriority::FIRST),
                false => assert_eq!(candidate_priority, CandidatePriority::LAST),
            };
            for cube in MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw) {
                placed(cube.priority(&key), cube.blueprint(&key).is_some());
            }
        }
        let cpu = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &devices::cpu_12())
            .swap_remove(0);
        assert_eq!(cpu.blueprint(&key), None);
        let rdna = offering(devices::rdna_range());
        let lead = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &rdna).swap_remove(0);
        assert!(lead.blueprint(&key).is_some());
    }

    /// A batched key prepares a batched launch: the batch is distributed to cubes, not dropped.
    #[test]
    fn a_batched_key_distributes_its_batch() {
        let problem = MatmulRoutineProblem::from(&key(8, 256, 256, 256).shape());
        assert_eq!(problem.batches, [8]);
        let hw = offering(devices::strix_halo());
        let lead = MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, &hw).swap_remove(0);
        let single = lead.blueprint(&key(1, 256, 256, 256)).expect("runs here");
        let batched = lead.blueprint(&key(8, 256, 256, 256)).expect("runs here");
        assert_ne!(single.0.partitioning, batched.0.partitioning);
    }

    /// The few-rows table's candidates: the shuffle routines, every quantized weight launched as stored.
    mod few_rows {
        use super::*;
        use crate::kernels::CubeShuffleMatmulRoutine;
        use crate::kernels::PlaneShuffleMatmulRoutine;
        use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
        use cubecl::tile::Axis;
        use cubek_std::devices;

        /// A key over an `f16` product of `batch` batches of `m` rows against a `k × n` weight whose
        /// buffer runs along `rhs`, or along neither axis for `None`.
        fn key(batch: usize, m: usize, n: usize, k: usize, rhs: Option<Axis>) -> MatmulTuneKey {
            MatmulTuneKey::new(
                F16,
                k,
                n,
                m,
                batch,
                OperandKey::float(Some(K), MatmulBatching::new(&[batch])),
                OperandKey::float(rhs, MatmulBatching::Shared),
            )
        }

        /// A key over `m` rows of `f16` against a `k × n` q4 weight packed along `K`.
        fn quantized_key(m: usize, n: usize, k: usize) -> MatmulTuneKey {
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q4S)
                .per_block([32, 1], ScaleDtype::F16)
                .with_store(QuantStore::PackedU32(1));
            MatmulTuneKey::new(
                F16,
                k,
                n,
                m,
                1,
                OperandKey::float(Some(K), MatmulBatching::Shared),
                OperandKey {
                    kind: OperandKind::Quantized(scheme),
                    ..OperandKey::float(Some(K), MatmulBatching::Shared)
                },
            )
        }

        #[test]
        fn a_candidate_per_setting_the_default_first() {
            let hw = devices::strix_halo();
            let listed: Vec<_> = MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw)
                .iter()
                .map(|candidate| *candidate.strategy())
                .collect();
            assert_eq!(listed, PlaneShuffleMatmulRoutine.settings(&hw));
        }

        /// A key's batches are read as rows against one weight, as the whole-rows tensors merge an
        /// activation's leading dims.
        #[test]
        fn a_keys_batches_are_read_as_rows() {
            let plane =
                MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &devices::strix_halo())
                    .swap_remove(0);
            let (blueprint, _) = plane
                .blueprint(&key(2, 4, 2048, 2048, Some(K)))
                .expect("eight rows against a col weight run here");
            assert_eq!(blueprint.microkernel.rows, 8);
        }

        /// Where no shuffle routine serves such a product, or the device runs none of it, a candidate
        /// is tried last and claims no launch: a quantized weight, a weight lined along neither axis,
        /// more rows than a work item holds, a device that commits to no plane width.
        #[test]
        fn a_candidate_serving_nothing_is_tried_last_and_claims_no_launch() {
            let cases = [
                (devices::strix_halo(), quantized_key(1, 2048, 2048)),
                (devices::strix_halo(), key(1, 1, 2048, 2048, None)),
                (devices::strix_halo(), key(1, 16, 2048, 2048, Some(K))),
                (devices::wgpu_no_subgroups(), key(1, 1, 2048, 2048, Some(K))),
            ];
            for (hw, key) in cases {
                for plane in MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw) {
                    let settings = plane.strategy();
                    assert_eq!(plane.priority(&key), CandidatePriority::LAST, "{settings}");
                    assert_eq!(plane.blueprint(&key), None, "{settings}");
                }
                for cube in MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, &hw) {
                    let settings = cube.strategy();
                    assert_eq!(cube.priority(&key), CandidatePriority::LAST, "{settings}");
                    assert_eq!(cube.blueprint(&key), None, "{settings}");
                }
            }
        }

        /// A candidate is placed by the one rule: a cube's run of columns tiles a decode projection,
        /// so both routines are tried first.
        #[test]
        fn the_priority_reads_the_block_one_cube_covers() {
            let hw = devices::strix_halo();
            let key = key(1, 1, 2048, 2048, Some(K));
            let plane = MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, &hw).swap_remove(0);
            let cube = MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, &hw).swap_remove(0);
            assert_eq!(plane.priority(&key), CandidatePriority::FIRST);
            assert_eq!(cube.priority(&key), CandidatePriority::FIRST);
        }
    }

    /// The many-rows table's routines: their estimates on a strix halo.
    mod many_rows_estimates {
        use crate::kernels::CubeStagedMatmulRoutine;
        use crate::kernels::UnitRegistersMatmulRoutine;
        use crate::kernels::{MatmulRoutine, MatmulRoutineProblem};
        use cubecl::features::MmaConfig;
        use cubecl::ir::{ElemType, FloatKind};
        use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel};
        use cubek_std::Hardware;
        use cubek_std::devices;
        use cubek_std::microkernel::KernelEstimate;
        use cubek_std::routine::EstimatedProblem;

        const F16: ElemType = ElemType::Float(FloatKind::F16);
        const F32: ElemType = ElemType::Float(FloatKind::F32);

        /// The dev machine's Radeon 8060S offering the `16×16×16` `f16` instruction summed in `f32`.
        fn strix_halo() -> Hardware {
            devices::offering_cmma(
                devices::strix_halo(),
                [MmaConfig {
                    a_type: F16,
                    b_type: F16,
                    cd_type: F32,
                    m: 16,
                    n: 16,
                    k: 16,
                }],
            )
        }

        /// Every part of `routine`'s estimate at every setting it lists, on an aligned product and
        /// a ragged one: each owner set its own, and the cubes perform at least what the problem
        /// asks.
        fn assert_every_estimate_is_whole<M: MatmulMicrokernel, SM, R: MatmulRoutine<M, SM>>(
            routine: R,
        ) {
            let hw = &strix_halo();
            let mut prepared = 0;
            for settings in routine.settings(hw) {
                for (m, n, k) in [(2048, 2048, 6144), (37, 29, 19)] {
                    let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
                    let Ok((blueprint, grid)) = routine.prepare(hw, &problem, &settings) else {
                        continue;
                    };
                    prepared += 1;
                    let mut estimate = KernelEstimate::from(&grid);
                    routine.estimate(&problem, &blueprint, &grid, &mut estimate);
                    let (staging, cubes) = (estimate.staging.as_ref(), estimate.cubes);
                    assert!(estimate.work.is_some(), "{settings}: no work");
                    let staging = staging.unwrap_or_else(|| panic!("{settings}: no staging"));
                    assert!(
                        staging.stages > 0 && !staging.reads.is_empty(),
                        "{settings}"
                    );
                    let performed = staging.operations * cubes.expect("a static grid") as f64;
                    let useful = problem.demand().compute_ops as f64;
                    assert!(performed >= useful, "{settings}: {performed} < {useful}");
                }
            }
            assert!(prepared > 0, "no setting of the routine prepares here");
        }

        #[test]
        fn every_routine_estimates_every_setting_whole() {
            assert_every_estimate_is_whole(UnitRegistersMatmulRoutine);
            assert_every_estimate_is_whole(CubeStagedMatmulRoutine);
            assert_every_estimate_is_whole(crate::kernels::HypercubeRelayMatmulRoutine);
            assert_every_estimate_is_whole(crate::kernels::HypercubeLastArrivalMatmulRoutine);
        }
    }

    /// The few-rows table's routines: their estimates on an rtx2060.
    mod few_rows_estimates {
        use crate::kernels::CubeShuffleMatmulRoutine;
        use crate::kernels::PlaneShuffleMatmulRoutine;
        use crate::kernels::{MatmulRoutine, MatmulRoutineProblem};
        use cubecl::ir::{ElemType, FloatKind};
        use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel};
        use cubek_std::Hardware;
        use cubek_std::devices;
        use cubek_std::microkernel::KernelEstimate;
        use cubek_std::routine::EstimatedProblem;

        const F16: ElemType = ElemType::Float(FloatKind::F16);

        /// The bench machine's rtx2060, roughly: thirty compute units, 64 KiB of shared memory,
        /// planes of 32, vectors of eight, no TMA.
        fn rtx2060() -> Hardware {
            let mut hw = devices::h100();
            let topology = devices::topology(&mut hw);
            topology.num_streaming_multiprocessors = Some(30);
            topology.max_shared_memory_size = 64 * 1024;
            topology.max_vector_size = 8;
            topology.num_tensor_cores = None;
            devices::features(&mut hw).tma.clear();
            hw
        }

        /// Every part of `routine`'s estimate at every setting it lists, on decode projections and
        /// a ragged product: each owner set its own, and the cubes perform at least what the
        /// problem asks.
        fn assert_every_estimate_is_whole<Mk: MatmulMicrokernel, R: MatmulRoutine<Mk>>(routine: R) {
            let hw = &rtx2060();
            let mut prepared = 0;
            for settings in routine.settings(hw) {
                for (m, n, k) in [(1, 2048, 6144), (8, 4096, 2048), (3, 29, 1024)] {
                    for rhs_layout in [MatmulLayout::Col, MatmulLayout::Row] {
                        let problem = MatmulRoutineProblem::dense(m, n, k, F16, rhs_layout);
                        let Ok((blueprint, grid)) = routine.prepare(hw, &problem, &settings) else {
                            continue;
                        };
                        prepared += 1;
                        let mut estimate = KernelEstimate::from(&grid);
                        routine.estimate(&problem, &blueprint, &grid, &mut estimate);
                        let (staging, cubes) = (estimate.staging.as_ref(), estimate.cubes);
                        assert!(estimate.work.is_some(), "{settings}: no work");
                        let staging = staging.unwrap_or_else(|| panic!("{settings}: no staging"));
                        assert!(
                            staging.stages > 0 && !staging.reads.is_empty(),
                            "{settings}"
                        );
                        let performed = staging.operations * cubes.expect("a static grid") as f64;
                        let useful = problem.demand().compute_ops as f64;
                        assert!(performed >= useful, "{settings}: {performed} < {useful}");
                    }
                }
            }
            assert!(prepared > 0, "no setting of the routine prepares here");
        }

        #[test]
        fn every_routine_estimates_every_setting_whole() {
            assert_every_estimate_is_whole(PlaneShuffleMatmulRoutine);
            assert_every_estimate_is_whole(CubeShuffleMatmulRoutine);
        }
    }
}
