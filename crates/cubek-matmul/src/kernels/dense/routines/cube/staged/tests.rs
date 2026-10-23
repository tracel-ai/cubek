//! Tests for the staged cube matmul routine: its settings, what each prepares, and the refusals.

mod base {
    use super::super::*;
    use crate::kernels::CUBE_STRIPS;
    use crate::kernels::{MatmulRoutineProblem, MatmulView};
    use core::num::NonZeroUsize;
    use cubecl::features::MmaConfig;
    use cubecl::features::{WgmmaConfig, WgmmaElems};
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::tile::Scratch;
    use cubecl::tile::StageStorage;
    use cubecl::tile::launch::Refusal;
    use cubecl::tile::layout::ElemAlignment;
    use cubecl::tile::{launch::Delivery, space::CubeOrder, stage::RowChunks};
    use cubek_microkernel::matmul::MatmulOperand;
    use cubek_microkernel::matmul::MatmulPartitioning;
    use cubek_microkernel::matmul::dense::cube::{
        CubeStagedDelivery, CubeStagedMicrokernel, CubeStagedMicrokernelSettings, CubeStagedRows,
        CubeStagedStore,
    };
    use cubek_microkernel::matmul::dense::cube::{
        CubeStagedFill, CubeStagedFillingRefusal, CubeStagedTmaRefusal,
    };
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentMicrokernelSettings;
    use cubek_microkernel::matmul::dense::plane::PlaneFragmentTransport;
    use cubek_microkernel::matmul::dense::plane::{
        PlaneFragmentInstruction, PlaneFragmentMicrokernel, PlaneFragmentStepSum,
    };
    use cubek_microkernel::matmul::staged_walk::MatmulPlaneLayout;
    use cubek_microkernel::matmul::{MatmulDestination, MatmulLayout, MatmulOperands, MatmulOrder};
    use cubek_microkernel::staging::StageRing;
    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::matmul_axes::{K, M, N};
    use cubek_std::microkernel::Microkernel;
    use cubek_std::routine::Routine;

    /// Why `hw` cannot deliver `problem`'s stages under `partitioning` by the TMA engine.
    fn tma_refusal(
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        partitioning: &MatmulPartitioning,
        storage: &StageStorage,
        ring: &StageRing,
        plane: &PlaneFragmentMicrokernel,
    ) -> Option<CubeStagedTmaRefusal> {
        CubeStagedTmaRefusal::new(
            hw,
            &MatmulView::of::<CubeStagedMicrokernel>(problem)
                .expect(MatmulView::PREPARED)
                .work_item(),
            &problem.buffers(plane.orders),
            partitioning,
            &partitioning.stage(),
            storage,
            ring,
            plane,
        )
    }

    /// The stage a box would land for `blueprint`'s instruction, whatever its fill resolved to.
    fn landed(blueprint: &CubeStagedMatmulRoutineBlueprint) -> StageStorage {
        blueprint
            .microkernel
            .plane
            .instruction
            .landed_stage_storage(
                blueprint.partitioning.batch_axes(),
                cubek_microkernel::matmul::MatmulAxes::MNK,
                blueprint.partitioning.stage().extent(K),
            )
    }
    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A device offering the `16×16×16` `f16` instruction summed in `f32`.
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

    fn problem() -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(37, 29, 19, F16, MatmulLayout::Row)
    }

    /// The default, its microkernel's settings as `set` leaves them.
    fn default_with(
        set: impl FnOnce(&mut CubeStagedMicrokernelSettings),
    ) -> CubeStagedMatmulRoutineSettings {
        let mut settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        set(&mut settings.microkernel);
        settings
    }

    /// [`SPLIT_ALONG_K`](CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K), its microkernel's
    /// settings as `set` leaves them.
    fn split_with(
        set: impl FnOnce(&mut CubeStagedMicrokernelSettings),
    ) -> CubeStagedMatmulRoutineSettings {
        let mut settings = CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K;
        set(&mut settings.microkernel);
        settings
    }

    /// Every setting prepares on a row and a col weight, padded stages and all, but a ring whose
    /// stages the device's shared memory cannot hold; the default always prepares.
    #[test]
    fn every_setting_prepares_where_the_instruction_is_offered() {
        let hw = offering(devices::strix_halo());
        for problem in [problem(), col_weight()] {
            for (position, settings) in CubeStagedMatmulRoutine.settings(&hw).iter().enumerate() {
                let prepared = CubeStagedMatmulRoutine.prepare(&hw, &problem, settings);
                let outgrown = matches!(prepared, Err(Refusal::SharedMemoryPastDevice { .. }));
                assert!(
                    prepared.is_ok() || (position > 0 && outgrown),
                    "{settings}: {prepared:?}"
                );
            }
        }
    }

    /// The deep box on eight planes stages 40 KB padded: a second stage in flight is past a core
    /// of 64 KB of shared memory, one stage with its next waiting in registers is not, and a core
    /// of 128 KB holds either, which is why both are listed.
    #[test]
    fn the_deep_box_rings_one_stage_where_two_outgrow_the_shared_memory() {
        let count = |n: usize| NonZeroUsize::new(n).unwrap();
        let deep = |ring| {
            default_with(|microkernel| {
                microkernel.plane = PlaneFragmentMicrokernelSettings {
                    along_m: count(4),
                    along_n: count(4),
                    depth: count(2),
                    ..microkernel.plane
                };
                microkernel.ring = ring;
            })
        };
        let one_stage = deep(StageRing::SINGLE_THROUGH_REGISTERS);
        let two_stages = deep(StageRing::DOUBLE_THROUGH_REGISTERS);
        let small = offering(devices::strix_halo());
        let large = {
            let mut hw = small.clone();
            devices::topology(&mut hw).max_shared_memory_size = 128 * 1024;
            hw
        };
        let prepared =
            |hw: &Hardware, settings| CubeStagedMatmulRoutine.prepare(hw, &col_weight(), &settings);
        assert!(prepared(&small, one_stage).is_ok());
        assert!(matches!(
            prepared(&small, two_stages),
            Err(Refusal::SharedMemoryPastDevice { .. })
        ));
        assert!(prepared(&large, one_stage).is_ok());
        assert!(prepared(&large, two_stages).is_ok());
    }

    /// A setting settles to one launch at an anchored row count and at a row count the key
    /// anchors to it: the tuner ranks a candidate on the first and launches it on the second, so
    /// what one cube holds, its shape, its ring and its fill are the same for both, and so is a
    /// refusal. A box sixteen fragments high cuts a tile of 244 rows short and none of 256, and a
    /// tile cut short is where its fragments bounce through a window.
    #[test]
    fn a_setting_settles_alike_at_the_rows_a_key_anchors() {
        use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
        let hw = offering(devices::strix_halo());
        let shape = |m, n, k, rhs| MatmulShape {
            m,
            n,
            k,
            batch: 1,
            elem: F16,
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(rhs), MatmulBatching::Shared),
        };
        let ranked = |problem: &MatmulRoutineProblem,
                      settings: &CubeStagedMatmulRoutineSettings| {
            CubeStagedMatmulRoutine
                .prepare(&hw, problem, settings)
                .map(|(blueprint, grid)| {
                    (
                        grid.shared_memory,
                        grid.launch.cube_dim,
                        blueprint.partitioning.cube_box(),
                        blueprint.microkernel.ring,
                        blueprint.microkernel.fill,
                    )
                })
                .map_err(|refusal| format!("{refusal:?}"))
        };
        for (n, k, rhs) in [(4096, 1024, N), (1024, 3072, K)] {
            let anchored = MatmulRoutineProblem::from(&shape(256, n, k, rhs));
            let short = MatmulRoutineProblem::from(&shape(244, n, k, rhs));
            let default = CubeStagedMatmulRoutineSettings::DEFAULT;
            assert!(
                ranked(&short, &default).is_ok(),
                "the default on n {n}, k {k}"
            );
            for settings in CubeStagedMatmulRoutine.settings(&hw) {
                assert_eq!(
                    ranked(&anchored, &settings),
                    ranked(&short, &settings),
                    "{settings} on n {n}, k {k}"
                );
            }
        }
    }

    /// The `16×16×16` `f16` shape summed in `f32`.
    const SHAPE: MmaConfig = MmaConfig {
        a_type: F16,
        b_type: F16,
        cd_type: F32,
        m: 16,
        n: 16,
        k: 16,
    };

    /// The instruction the default elects on `hw` for `problem`, and the width its planes launch.
    fn elected(hw: Hardware, problem: &MatmulRoutineProblem) -> (PlaneFragmentInstruction, usize) {
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&hw, problem, &settings)
            .expect("the plane routine prepares everywhere");
        (blueprint.microkernel.plane.instruction, grid.planes().width)
    }

    /// The vendor's transport, then the tile DSL's for the same shape, then the software
    /// instruction.
    #[test]
    fn the_candidates_are_the_matrix_instructions_then_the_software_one() {
        let hw = devices::offering_mma(offering(devices::strix_halo()), [SHAPE]);
        let item = MatmulView::of::<CubeStagedMicrokernel>(&problem())
            .expect(MatmulView::PREPARED)
            .work_item();
        let settings = PlaneFragmentMicrokernelSettings::DEFAULT;
        let instructions: Vec<_> = PlaneFragmentMicrokernel::candidates(&hw, &item, &settings)
            .map(|candidate| candidate.instruction)
            .collect();
        assert_eq!(
            instructions,
            [
                PlaneFragmentInstruction::Cmma { shape: SHAPE },
                PlaneFragmentInstruction::Mma { shape: SHAPE },
                PlaneFragmentInstruction::Registers {
                    step_sum: PlaneFragmentStepSum::Wide,
                },
            ]
        );
    }

    /// A setting asking for the tile DSL's transport first reaches the shape through it before the
    /// vendor's, and still ends in the software instruction.
    #[test]
    fn a_transport_setting_orders_the_two_transports() {
        let hw = devices::offering_mma(offering(devices::strix_halo()), [SHAPE]);
        let item = MatmulView::of::<CubeStagedMicrokernel>(&problem())
            .expect(MatmulView::PREPARED)
            .work_item();
        let settings = PlaneFragmentMicrokernelSettings {
            transport: PlaneFragmentTransport::Mma,
            ..PlaneFragmentMicrokernelSettings::DEFAULT
        };
        let instructions: Vec<_> = PlaneFragmentMicrokernel::candidates(&hw, &item, &settings)
            .map(|candidate| candidate.instruction)
            .collect();
        assert_eq!(
            instructions,
            [
                PlaneFragmentInstruction::Mma { shape: SHAPE },
                PlaneFragmentInstruction::Cmma { shape: SHAPE },
                PlaneFragmentInstruction::Registers {
                    step_sum: PlaneFragmentStepSum::Wide,
                },
            ]
        );
    }

    /// A device offering its shape only under the tile DSL's transport runs it there.
    #[test]
    fn a_shape_offered_only_to_mma_runs_through_it() {
        let hw = devices::offering_mma(devices::strix_halo(), [SHAPE]);
        let (instruction, width) = elected(hw, &problem());
        assert!(
            matches!(instruction, PlaneFragmentInstruction::Mma { .. }),
            "{instruction}"
        );
        assert_eq!(width, 32);
    }

    /// Hopper's warpgroup MMA of `f16` into `f32`: `64 × n × 16`, `n` a multiple of 8 to 256.
    const WGMMA_F16: WgmmaConfig = WgmmaConfig {
        elems: WgmmaElems {
            a: F16,
            b: F16,
            cd: F32,
        },
        m: 64,
        n_granularity: 8,
        n_max: 256,
        k: 16,
    };

    /// An H100 offering its warpgroup MMA beside a fragment instruction.
    fn hopper() -> Hardware {
        devices::offering_wgmma(offering(devices::h100()), [WGMMA_F16])
    }

    /// On a device with a warpgroup MMA, only the setting that asks for it runs it, on a col
    /// weight, and declines a row weight it cannot read; every other setting prepares as it does
    /// without one, on any other instruction.
    #[test]
    fn only_the_setting_that_asks_for_it_runs_the_warpgroup_mma() {
        let hw = hopper();
        let settings = CubeStagedMatmulRoutine.settings(&hw);
        let asks = |settings: &CubeStagedMatmulRoutineSettings| {
            settings.microkernel.plane.transport == PlaneFragmentTransport::Wgmma
        };
        assert_eq!(settings.iter().filter(|s| asks(s)).count(), 1);
        for (problem, col) in [(problem(), false), (col_weight(), true)] {
            for (position, settings) in settings.iter().enumerate() {
                let prepared = CubeStagedMatmulRoutine.prepare(&hw, &problem, settings);
                match (asks(settings), col, prepared) {
                    (true, true, Ok((blueprint, _))) => assert!(matches!(
                        blueprint.microkernel.plane.instruction,
                        PlaneFragmentInstruction::Wgmma { n: 128, .. }
                    )),
                    (true, false, Err(Refusal::Declined { .. })) => {}
                    (false, _, Ok((blueprint, _))) => assert!(
                        !matches!(
                            blueprint.microkernel.plane.instruction,
                            PlaneFragmentInstruction::Wgmma { .. }
                        ),
                        "{settings}"
                    ),
                    (false, _, Err(Refusal::SharedMemoryPastDevice { .. })) if position > 0 => {}
                    (_, _, prepared) => panic!("{settings} on a col weight {col}: {prepared:?}"),
                }
            }
        }
    }

    /// A `128 × 256 × 512` `f16` product whose rhs is a col weight: an `[n, k]` buffer behind the
    /// `[k, n]` view.
    fn col_weight() -> MatmulRoutineProblem {
        use crate::kernels::tune::key::{MatmulBatching, MatmulShape, OperandKey};
        MatmulRoutineProblem::from(&MatmulShape {
            m: 128,
            n: 256,
            k: 512,
            batch: 1,
            elem: F16,
            lhs: OperandKey::float(Some(K), MatmulBatching::Batched),
            rhs: OperandKey::float(Some(K), MatmulBatching::Shared),
        })
    }

    /// Both matrix transports load their `B` col-major, so a col weight is bound in its buffer's
    /// order and read in lines along `K`; the software instruction reads the order the product
    /// names, so the same weight is read one element at a time.
    #[test]
    fn the_matrix_transports_bind_a_col_weight_in_its_buffers_order() {
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let prepared = |hw: Hardware| {
            CubeStagedMatmulRoutine
                .prepare(&hw, &col_weight(), &settings)
                .expect("the routine runs here")
                .0
        };
        let cmma = prepared(offering(devices::strix_halo()));
        let mma = prepared(devices::offering_mma(devices::strix_halo(), [SHAPE]));
        for blueprint in [cmma, mma] {
            assert_eq!(
                blueprint.microkernel.plane.orders.rhs,
                MatmulOrder::InBufferOrder,
                "{}",
                blueprint.microkernel.plane.instruction
            );
            assert!(blueprint.widths.rhs > 1, "{:?}", blueprint.widths);
        }

        let software = prepared(devices::strix_halo());
        let no_subgroups = prepared(offering(devices::wgpu_no_subgroups()));
        for blueprint in [software, no_subgroups] {
            assert_eq!(
                blueprint.microkernel.plane.orders.rhs,
                MatmulOrder::Stated,
                "{}",
                blueprint.microkernel.plane.instruction
            );
            assert_eq!(blueprint.widths.rhs, 1);
        }
    }

    /// The software instruction's register block lines its accumulator with the rhs, so under a
    /// col weight read one element at a time the output is written one element at a time too.
    #[test]
    fn the_software_instruction_lines_its_output_with_the_rhs() {
        let (blueprint, _) = CubeStagedMatmulRoutine
            .prepare(
                &devices::strix_halo(),
                &col_weight(),
                &CubeStagedMatmulRoutineSettings::DEFAULT,
            )
            .expect("the software instruction runs everywhere");
        assert_eq!((blueprint.widths.rhs, blueprint.widths.out), (1, 1));
    }

    /// A row weight is bound as the product names it whatever the instruction.
    #[test]
    fn a_row_weight_is_bound_as_the_product_names_it() {
        let problem = MatmulRoutineProblem::dense(128, 256, 512, F16, MatmulLayout::Row);
        for hw in [
            offering(devices::strix_halo()),
            offering(devices::rdna_range()),
        ] {
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &CubeStagedMatmulRoutineSettings::DEFAULT)
                .expect("the routine runs here");
            assert_eq!(blueprint.microkernel.plane.orders.rhs, MatmulOrder::Stated);
        }
    }

    /// No instruction for the elements, or a plane whose width the device will not commit to, which
    /// a fragment spread across the plane needs: the software instruction runs, on its own units.
    #[test]
    fn the_software_instruction_runs_where_no_matrix_instruction_does() {
        let software = PlaneFragmentInstruction::Registers {
            step_sum: PlaneFragmentStepSum::Wide,
        };
        let units = software.units(&devices::strix_halo()).unwrap() as usize;
        for hw in [
            devices::strix_halo(),
            offering(devices::rdna_range()),
            offering(devices::wgpu_no_subgroups()),
        ] {
            assert_eq!(elected(hw.clone(), &problem()), (software, units), "{hw:?}");
        }
    }

    /// A narrow step changes what the product computes, so it is an opt-in: no setting the routine
    /// lists, on any device, asks for one.
    #[test]
    fn no_listed_setting_steps_narrow() {
        for hw in [
            devices::strix_halo(),
            devices::wgpu_no_subgroups(),
            devices::h100(),
        ] {
            for settings in CubeStagedMatmulRoutine.settings(&hw) {
                assert_eq!(
                    settings.microkernel.plane.step_sum,
                    PlaneFragmentStepSum::Wide,
                    "{settings}"
                );
            }
        }
    }

    /// The device prunes the list: the engine's twins are listed only where the copy engine runs,
    /// since elsewhere each would resolve to the loaded setting already listed beside it.
    #[test]
    fn the_engine_twins_are_listed_only_where_the_copy_engine_runs() {
        let engine_twins = |hw: Hardware| {
            CubeStagedMatmulRoutine
                .settings(&hw)
                .into_iter()
                .filter(|settings| {
                    settings.microkernel.fill.delivery == CubeStagedDelivery::AsyncCopy
                })
                .count()
        };
        assert!(engine_twins(devices::h100()) > 0);
        for hw in [devices::strix_halo(), devices::wgpu_no_subgroups()] {
            assert_eq!(engine_twins(hw.clone()), 0, "{hw:?}");
        }
    }

    /// A narrow step where the setting asks for it and the operands admit it — `f16` summed in
    /// `f32` — and the wide step's kernel everywhere else, matrix instructions included.
    #[test]
    fn a_narrow_step_is_the_software_instructions_on_f16_alone() {
        let narrow = default_with(|microkernel| {
            microkernel.plane.step_sum = PlaneFragmentStepSum::Narrow;
        });
        let elected = |hw: &Hardware, element| {
            let problem = MatmulRoutineProblem::dense(37, 29, 19, element, MatmulLayout::Row);
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(hw, &problem, &narrow)
                .expect("the plane routine prepares here");
            blueprint.microkernel.plane.instruction.step_sum()
        };
        let software = devices::strix_halo();
        assert_eq!(elected(&software, F16), PlaneFragmentStepSum::Narrow);
        assert_eq!(elected(&software, F32), PlaneFragmentStepSum::Wide);
        assert_eq!(
            elected(&offering(devices::strix_halo()), F16),
            PlaneFragmentStepSum::Wide
        );
    }

    /// A CPU runtime holds a dozen units a cube, fewer than one cube of planes of the software
    /// instruction: the plane routine declines there, and the unit routine is the floor.
    #[test]
    fn declines_a_cube_past_the_units_a_cpu_holds() {
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let declined = CubeStagedMatmulRoutine.prepare(&devices::cpu_12(), &problem(), &settings);
        assert!(
            matches!(declined, Err(Refusal::CubePastDevice { .. })),
            "{declined:?}"
        );
    }

    /// The default prepares on devices with no matrix instruction, ragged shapes included; a
    /// setting other than the default may ask for a cube the device cannot hold, and declines
    /// saying so, and one splitting `K` declines where shared memory takes no atomic add.
    #[test]
    fn every_setting_prepares_on_the_software_instruction() {
        for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
            for (position, settings) in CubeStagedMatmulRoutine.settings(&hw).iter().enumerate() {
                for (m, n, k) in [(1, 128, 256), (37, 29, 19), (512, 4096, 4096)] {
                    let problem = MatmulRoutineProblem::dense(m, n, k, F32, MatmulLayout::Row);
                    let prepared = CubeStagedMatmulRoutine.prepare(&hw, &problem, settings);
                    let outgrown = matches!(
                        prepared,
                        Err(Refusal::SharedMemoryPastDevice { .. } | Refusal::CubePastDevice { .. })
                    );
                    let unmerged = settings.microkernel.planes_along_k.get() > 1
                        && !hw.shared_f32_atomic_add()
                        && matches!(
                            prepared,
                            Err(Refusal::Declined {
                                why: CubeStagedMicrokernel::DECLINE
                            })
                        );
                    assert!(
                        prepared.is_ok() || (position > 0 && (outgrown || unmerged)),
                        "{settings} on {hw:?}: {prepared:?}"
                    );
                }
            }
        }
    }

    /// The tile DSL's multi-row box on the `16×16×16` instruction: four planes stacked along `m`,
    /// each holding two rows of eight fragments, so one plane owns `32 × 128` and the cube's box is
    /// `128 × 128`, walked two instruction steps of `K` a stage, the cubes in strips along `m`.
    #[test]
    fn the_multi_row_box_stacks_its_planes_along_m() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(1024, 1024, 256, F16, MatmulLayout::Row);
        let multi_row = CubeStagedMatmulRoutine
            .settings(&hw)
            .into_iter()
            .find(|settings| settings.microkernel.planes == MatmulPlaneLayout::new(4, 1))
            .expect("the list holds a multi-row box");
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&hw, &problem, &multi_row)
            .expect("the routine runs here");
        let (cube, stage) = (
            blueprint.partitioning.cube_box(),
            blueprint.partitioning.stage(),
        );
        assert_eq!((cube.extent(M), cube.extent(N)), (128, 128));
        assert_eq!(stage.extent(K), 32);
        assert_eq!(blueprint.microkernel.plane.tile_extent(M), 32);
        assert_eq!(grid.planes().count, 4);
        assert_eq!(
            blueprint.partitioning.cube_order(),
            CubeOrder::SwizzleRow(4)
        );
    }

    /// A second stage in flight holds a second stage of shared memory, and the blueprint carries
    /// the depth to the kernel's ring.
    #[test]
    fn a_double_ring_holds_two_stages() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let prepared = |ring| {
            let settings = default_with(|microkernel| microkernel.ring = ring);
            CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the routine runs here")
        };
        let (single, single_grid) = prepared(StageRing::SINGLE);
        let (double, double_grid) = prepared(StageRing::DOUBLE);
        let stage = single.partitioning.stage().bytes(
            &hw,
            problem.elements,
            &single.shared_memory,
            single.microkernel.plane.orders,
        );
        assert_eq!(double_grid.shared_memory - single_grid.shared_memory, stage);
        assert_eq!(double.microkernel.ring, StageRing::DOUBLE);
        assert_eq!(single.microkernel.ring, StageRing::SINGLE);
    }

    /// A ring prefetching into registers holds each unit's share of a stage there: past the tile
    /// DSL's bound it prefetches into its slots instead, and a ring prefetching into its slots is
    /// left as it is.
    #[test]
    fn a_register_prefetch_resolves_to_the_slots_past_a_units_registers() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&hw, &problem, &CubeStagedMatmulRoutineSettings::DEFAULT)
            .expect("the routine runs here");
        let registers = StageRing::DOUBLE_THROUGH_REGISTERS;
        let partitioning = &blueprint.partitioning;
        let planes = grid.planes();
        let cube_units = planes.count * planes.width;
        let one_unit = 1;
        let item = MatmulView::of::<CubeStagedMicrokernel>(&problem)
            .expect(MatmulView::PREPARED)
            .work_item();
        let operands = partitioning
            .stage()
            .operands(item.elements, blueprint.microkernel.plane.orders);
        assert_eq!(registers.resolve(&hw, &operands, cube_units), registers);
        assert_eq!(
            registers.resolve(&hw, &operands, one_unit),
            StageRing::DOUBLE
        );
        assert_eq!(
            StageRing::SINGLE.resolve(&hw, &operands, one_unit),
            StageRing::SINGLE
        );
    }

    /// Every listed setting prefetching into registers, where its ring keeps that prefetch, holds
    /// no more of both operands a unit than the tile DSL fetches, at the widths the blueprint
    /// settled: what the tile DSL checks at expansion, where a refusal is a launch that writes
    /// nothing. On devices of three plane widths, over a row weight, a col weight and a ragged
    /// problem.
    #[test]
    fn every_kept_register_prefetch_fits_what_cubek_fetches() {
        use cubecl::tile::stage::Prefetch;
        let mut kept = 0;
        for hw in [
            offering(devices::strix_halo()),
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
        ] {
            let prefetching = CubeStagedMatmulRoutine
                .settings(&hw)
                .into_iter()
                .filter(|settings| settings.microkernel.ring.prefetch == Prefetch::InRegisters);
            for settings in prefetching {
                for problem in [
                    MatmulRoutineProblem::dense(2048, 2048, 2048, F16, MatmulLayout::Row),
                    col_weight(),
                    problem(),
                ] {
                    let Ok((blueprint, grid)) =
                        CubeStagedMatmulRoutine.prepare(&hw, &problem, &settings)
                    else {
                        continue;
                    };
                    if blueprint.microkernel.ring.prefetch != Prefetch::InRegisters {
                        continue;
                    }
                    kept += 1;
                    let stage = blueprint.partitioning.stage();
                    let units = grid.planes().count * grid.planes().width;
                    let fetched =
                        |elements: usize, width: usize| Prefetch::scalars(elements, units, width);
                    let held = fetched(stage.extent(M) * stage.extent(K), blueprint.widths.lhs)
                        + fetched(stage.extent(K) * stage.extent(N), blueprint.widths.rhs);
                    assert!(
                        Prefetch::fits(held),
                        "{settings} on {hw:?}, {problem:?}: {held} scalars a unit"
                    );
                }
            }
        }
        assert!(kept > 0, "no listed setting kept its register prefetch");
    }

    /// A box asked of the TMA engine's store is stored by it wherever a descriptor addresses the
    /// output, each cube writing its own box once, and by the fragments everywhere else, so the
    /// setting runs on every problem. A box the engine stores is drained whole into shared memory
    /// and clipped by the descriptor, so its fragments never bounce at the output's edge.
    #[test]
    fn a_tma_store_is_taken_where_nothing_refuses_it() {
        let stored = |settings: CubeStagedMatmulRoutineSettings| CubeStagedMatmulRoutineSettings {
            microkernel: CubeStagedMicrokernelSettings {
                store: CubeStagedStore::Tma,
                ..settings.microkernel
            },
            ..settings
        };
        let settings = stored(default_with(|microkernel| {
            microkernel.ring = StageRing::DOUBLE
        }));
        let prepared = |hw: Hardware,
                        problem: &MatmulRoutineProblem,
                        settings: &CubeStagedMatmulRoutineSettings| {
            CubeStagedMatmulRoutine
                .prepare(&hw, problem, settings)
                .expect("the routine runs here")
                .0
        };
        let store = |hw: Hardware, problem: &MatmulRoutineProblem| {
            prepared(hw, problem, &settings).microkernel.store
        };
        let aligned = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let blueprint = prepared(devices::h100(), &aligned, &settings);
        assert_eq!(blueprint.microkernel.store, CubeStagedStore::Tma);
        assert_eq!(blueprint.widths.out, 1);

        let overhanging = MatmulRoutineProblem::dense(33, 512, 64, F16, MatmulLayout::Row);
        let blueprint = prepared(devices::h100(), &overhanging, &settings);
        assert_eq!(blueprint.microkernel.store, CubeStagedStore::Tma);
        assert_eq!(blueprint.microkernel.scratch, Scratch::None);

        // Rows of 60 halves are 120 bytes, not whole 16-byte lines.
        let misaligned = MatmulRoutineProblem::dense(256, 60, 64, F16, MatmulLayout::Row);
        let batched = MatmulRoutineProblem {
            batches: vec![2],
            ..aligned.clone()
        };
        for (hw, problem) in [
            (devices::strix_halo(), &aligned),
            (devices::h100(), &misaligned),
            (devices::h100(), &batched),
        ] {
            assert_eq!(
                store(hw, problem),
                CubeStagedStore::Fragments,
                "{problem:?}"
            );
        }
        let split = prepared(
            devices::h100(),
            &aligned,
            &stored(CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K),
        );
        assert_eq!(split.microkernel.store, CubeStagedStore::Fragments);
    }

    /// The TMA engine fills a stage where nothing refuses it, and both operands are read one
    /// element wide there; the units fill it everywhere else, under the refusal that names why.
    #[test]
    fn a_tma_fill_is_delivered_where_nothing_refuses_it() {
        let by_tma = CubeStagedFill {
            delivery: CubeStagedDelivery::Tma,
            filling_planes: 0,
        };
        // The software instruction's stage holds its cells as the operand lies, which a
        // descriptor's box lands whole.
        let software = devices::h100();
        let settings = default_with(|microkernel| {
            microkernel.ring = StageRing::DOUBLE;
            microkernel.fill = by_tma;
        });
        let prepared = |hw: &Hardware, problem: &MatmulRoutineProblem| {
            CubeStagedMatmulRoutine
                .prepare(hw, problem, &settings)
                .expect("the routine runs here")
                .0
        };
        let refusal = |hw: &Hardware, problem: &MatmulRoutineProblem| {
            let blueprint = prepared(hw, problem);
            tma_refusal(
                hw,
                problem,
                &blueprint.partitioning,
                &landed(&blueprint),
                &blueprint.microkernel.ring,
                &blueprint.microkernel.plane,
            )
        };
        let aligned = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let blueprint = prepared(&software, &aligned);
        assert_eq!(blueprint.microkernel.fill.delivery, CubeStagedDelivery::Tma);
        assert_eq!((blueprint.widths.lhs, blueprint.widths.rhs), (1, 1));
        let batched = MatmulRoutineProblem {
            batches: vec![2],
            ..aligned.clone()
        };
        let unaligned = MatmulRoutineProblem::dense(256, 512, 60, F16, MatmulLayout::Row);
        let sliced = MatmulRoutineProblem {
            start_bytes: MatmulOperands {
                lhs: 8,
                rhs: 0,
                out: 0,
            },
            ..aligned.clone()
        };
        // A weight stored in tiles no partitioning describes, read one element at a time: no
        // descriptor box addresses a tile, whatever its piece.
        let stored = MatmulRoutineProblem {
            elem_alignments: MatmulOperands {
                rhs: ElemAlignment::stored_piece(1),
                ..aligned.elem_alignments
            },
            buffer_alignments: MatmulOperands {
                rhs: ElemAlignment::stored_piece(1),
                ..aligned.buffer_alignments
            },
            ..aligned.clone()
        };
        assert_eq!(
            tma_refusal(
                &software,
                &aligned,
                &blueprint.partitioning,
                &blueprint.shared_memory,
                &StageRing::DOUBLE_THROUGH_REGISTERS,
                &blueprint.microkernel.plane
            ),
            Some(CubeStagedTmaRefusal::PrefetchInRegisters)
        );
        // A box lands its rows dense: never padded.
        let padded = StageStorage::Tiled {
            block: [M, N, K]
                .map(|axis| (axis, blueprint.partitioning.stage().extent(axis)))
                .to_vec(),
            chunks: RowChunks::Padded,
        };
        assert_eq!(
            tma_refusal(
                &software,
                &aligned,
                &blueprint.partitioning,
                &padded,
                &StageRing::DOUBLE,
                &blueprint.microkernel.plane
            ),
            Some(CubeStagedTmaRefusal::Unlanded(MatmulOperand::Lhs))
        );
        for (hw, problem, why) in [
            (
                devices::strix_halo(),
                &aligned,
                CubeStagedTmaRefusal::NoBulkTensorCopy,
            ),
            (software.clone(), &batched, CubeStagedTmaRefusal::Batched),
            (
                software.clone(),
                &unaligned,
                CubeStagedTmaRefusal::Misaligned(MatmulOperand::Lhs),
            ),
            (
                software.clone(),
                &sliced,
                CubeStagedTmaRefusal::UnalignedStart(MatmulOperand::Lhs),
            ),
            (
                software.clone(),
                &stored,
                CubeStagedTmaRefusal::StoredInTiles(MatmulOperand::Rhs),
            ),
            // A fragment reads a row weight's stage in blocks of its tile along `n`, which split
            // the rows a box lands whole.
            (
                offering(devices::h100()),
                &aligned,
                CubeStagedTmaRefusal::Unlanded(MatmulOperand::Rhs),
            ),
        ] {
            assert_eq!(refusal(&hw, problem), Some(why), "{problem:?}");
            assert_eq!(
                prepared(&hw, problem).microkernel.fill.delivery,
                CubeStagedDelivery::Copy
            );
        }
    }

    /// A col weight bound in its buffer's order stages whole rows of `K` under either matrix
    /// transport, so a box lands both operands' stages: in order for `cmma`, which reads a row off
    /// a pointer and a stride, swizzled as the descriptor swizzles them for the tile DSL's `mma`.
    #[test]
    fn a_col_weights_stages_land_by_tma_under_a_matrix_instruction() {
        let settings = default_with(|microkernel| {
            microkernel.ring = StageRing::DOUBLE;
            microkernel.fill = CubeStagedFill {
                delivery: CubeStagedDelivery::Tma,
                filling_planes: 0,
            };
        });
        let through_mma = devices::offering_mma(devices::h100(), [SHAPE]);
        for (hw, chunks) in [
            (offering(devices::h100()), RowChunks::InOrder),
            (through_mma, RowChunks::Swizzled),
        ] {
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &col_weight(), &settings)
                .expect("the routine runs here");
            assert_eq!(blueprint.microkernel.fill.delivery, CubeStagedDelivery::Tma);
            assert_eq!(
                blueprint.microkernel.plane.orders.rhs,
                MatmulOrder::InBufferOrder
            );
            let depth = blueprint.partitioning.stage().extent(K);
            assert!(
                matches!(
                    &blueprint.shared_memory,
                    StageStorage::Tiled { block, chunks: held }
                        if *held == chunks && block.contains(&(K, depth))
                ),
                "{:?}",
                blueprint.shared_memory
            );
        }
    }

    /// A cube's filling planes take no tile: launched beside them, its box, stage, leaf and
    /// overhang are its tiling's placement's, and only the planes it holds grow.
    #[test]
    fn filling_planes_widen_the_cube_and_nothing_else() {
        let settings = default_with(|microkernel| {
            microkernel.ring = StageRing::DOUBLE;
            microkernel.fill = CubeStagedFill {
                delivery: CubeStagedDelivery::Tma,
                filling_planes: 1,
            };
        });
        let problem = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let (blueprint, _) = CubeStagedMatmulRoutine
            .prepare(&devices::h100(), &problem, &settings)
            .expect("the routine runs here");
        assert_eq!(blueprint.microkernel.fill.filling_planes, 1);
        let placed = MatmulView::of::<CubeStagedMicrokernel>(&problem)
            .expect(MatmulView::PREPARED)
            .partitioning(
                blueprint.microkernel.tiling(),
                &[M, N],
                settings.cube_order,
                MatmulView::ONE_CUBE_A_BOX,
            );
        let launched = &blueprint.partitioning;
        assert_eq!(launched.cube_box(), placed.cube_box());
        assert_eq!(launched.stage(), placed.stage());
        assert_eq!(launched.leaf(), placed.leaf());
        assert_eq!(launched.overhangs_output(), placed.overhangs_output());
        assert_eq!(
            launched.as_tile().planes_per_cube(),
            placed.as_tile().planes_per_cube() + 1
        );
    }

    /// Filling planes are set aside where the TMA engine fills two stages in flight and no
    /// fragment bounces: the cube is one plane wider. None everywhere else.
    #[test]
    fn filling_planes_are_set_aside_only_beside_a_tma_fill() {
        let filled = CubeStagedFill {
            delivery: CubeStagedDelivery::Tma,
            filling_planes: 1,
        };
        let set_aside = |ring: StageRing, hw: Hardware, problem: &MatmulRoutineProblem| {
            let settings = default_with(|microkernel| {
                microkernel.ring = ring;
                microkernel.fill = filled;
            });
            let (blueprint, grid) = CubeStagedMatmulRoutine
                .prepare(&hw, problem, &settings)
                .expect("the routine runs here");
            (
                blueprint.microkernel.fill.filling_planes,
                grid.planes().count,
            )
        };
        let covered = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        let computing = CubeStagedMatmulRoutineSettings::DEFAULT
            .microkernel
            .planes
            .count();
        assert_eq!(
            set_aside(StageRing::DOUBLE, devices::h100(), &covered),
            (1, computing + 1)
        );
        for (ring, hw) in [
            (StageRing::DOUBLE, devices::strix_halo()),
            (StageRing::SINGLE, devices::h100()),
        ] {
            assert_eq!(set_aside(ring, hw, &covered), (0, computing), "{ring:?}");
        }
        assert_eq!(
            CubeStagedFillingRefusal::new(
                CubeStagedDelivery::Tma,
                &MatmulView::of::<CubeStagedMicrokernel>(&problem())
                    .expect(MatmulView::PREPARED)
                    .work_item(),
                &StageRing::DOUBLE,
                Scratch::OneTile,
                NonZeroUsize::MIN
            ),
            Some(CubeStagedFillingRefusal::Bounces)
        );
    }

    /// The units hand a stage to the copy engine where the device has one, the ring fills its
    /// slots and the lines are ones the engine copies whole; they load it everywhere else. A
    /// setting that asks for the engine therefore runs on every device, as its loaded twin where
    /// it cannot have it.
    #[test]
    fn a_stage_is_handed_to_the_copy_engine_only_where_it_can_be() {
        let engine = CubeStagedFill {
            delivery: CubeStagedDelivery::AsyncCopy,
            filling_planes: 0,
        };
        let delivery = |ring: StageRing, hw: Hardware, problem: &MatmulRoutineProblem| {
            let settings = default_with(|microkernel| {
                microkernel.ring = ring;
                microkernel.fill = engine;
            });
            CubeStagedMatmulRoutine
                .prepare(&offering(hw), problem, &settings)
                .expect("the routine runs here")
                .0
                .microkernel
                .fill
                .delivery
        };
        let covered = MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row);
        for ring in [StageRing::DOUBLE, StageRing::SINGLE] {
            assert_eq!(
                delivery(ring, devices::h100(), &covered),
                CubeStagedDelivery::AsyncCopy,
                "{ring:?}"
            );
        }
        for (ring, hw, problem) in [
            // The next stage waits in the units' registers, which the engine never fills.
            (
                StageRing::DOUBLE_THROUGH_REGISTERS,
                devices::h100(),
                &covered,
            ),
            // No copy engine.
            (StageRing::DOUBLE, devices::strix_halo(), &covered),
            // `k = 19` binds the lhs one `f16` wide: two bytes, which the engine does not copy.
            (StageRing::DOUBLE, devices::h100(), &problem()),
        ] {
            assert_eq!(
                delivery(ring, hw, problem),
                CubeStagedDelivery::Copy,
                "{ring:?} on {problem:?}"
            );
        }
        // Every unit fills, so none may be set aside; the operands state the engine's delivery
        // and a descriptor's state the loads, since tensor maps fill its stages.
        assert_eq!(
            CubeStagedFillingRefusal::new(
                CubeStagedDelivery::AsyncCopy,
                &MatmulView::of::<CubeStagedMicrokernel>(&problem())
                    .expect(MatmulView::PREPARED)
                    .work_item(),
                &StageRing::DOUBLE,
                Scratch::None,
                NonZeroUsize::MIN
            ),
            Some(CubeStagedFillingRefusal::CopiedByUnits)
        );
        assert_eq!(
            CubeStagedDelivery::AsyncCopy.bound(),
            Delivery::AsyncPerUnit
        );
        assert_eq!(CubeStagedDelivery::Tma.bound(), Delivery::SyncPerUnit);
    }

    /// A strip keeps the width its setting asks for whatever the cubes along its axis: the
    /// default's cube box is `128 × 256`, so `512 × 768` is four cubes along `m` and three along
    /// `n`, the strips along `n` one ragged strip.
    #[test]
    fn the_cubes_run_in_the_strips_their_setting_names() {
        let hw = offering(devices::strix_halo());
        let order = |cube_order| {
            let settings = CubeStagedMatmulRoutineSettings {
                cube_order,
                ..CubeStagedMatmulRoutineSettings::DEFAULT
            };
            let problem = MatmulRoutineProblem::dense(512, 768, 64, F16, MatmulLayout::Row);
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the routine runs here");
            assert_eq!(blueprint.partitioning.cubes(), (4, 3));
            blueprint.partitioning.cube_order()
        };
        let [along_m, along_n] = CUBE_STRIPS;
        assert_eq!(order(CubeOrder::RowMajor), CubeOrder::RowMajor);
        assert_eq!(order(along_m), CubeOrder::SwizzleRow(4));
        assert_eq!(order(along_n), CubeOrder::SwizzleCol(4));
    }

    /// The software instruction writes its cells on their own, so an output its tiles overhang
    /// holds no scratch.
    #[test]
    fn the_software_instruction_never_bounces() {
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let (blueprint, _) = CubeStagedMatmulRoutine
            .prepare(&devices::strix_halo(), &problem(), &settings)
            .expect("the software instruction runs here");
        assert_eq!(blueprint.microkernel.scratch, Scratch::None);
    }

    #[test]
    fn declines_a_cube_past_the_shared_memory() {
        let hw = {
            let mut hw = offering(devices::strix_halo());
            devices::topology(&mut hw).max_shared_memory_size = 1024;
            hw
        };
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let declined = CubeStagedMatmulRoutine.prepare(&hw, &problem(), &settings);
        assert!(
            matches!(declined, Err(Refusal::SharedMemoryPastDevice { .. })),
            "{declined:?}"
        );
    }

    /// An output its tiles cover exactly stores its fragments through their own intrinsic and
    /// holds no scratch; one its tiles overhang bounces through one.
    #[test]
    fn only_an_overhanging_output_drains_through_a_scratch() {
        let hw = offering(devices::strix_halo());
        let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
        let scratch = |m, n, k| {
            let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the default prepares here");
            blueprint.microkernel.scratch
        };
        assert_eq!(scratch(256, 512, 64), Scratch::None);
        assert_eq!(scratch(256, 512, 19), Scratch::None);
        assert_eq!(scratch(37, 29, 19), Scratch::OneTile);
    }

    /// A sink has no address for a fragment's store intrinsic, so an output its tiles cover
    /// exactly still bounces through a scratch when it is one.
    #[test]
    fn a_sink_drains_every_fragment_through_a_scratch() {
        let hw = offering(devices::strix_halo());
        let problem = MatmulRoutineProblem {
            destination: MatmulDestination::Sink,
            ..MatmulRoutineProblem::dense(256, 512, 64, F16, MatmulLayout::Row)
        };
        let (blueprint, _) = CubeStagedMatmulRoutine
            .prepare(&hw, &problem, &CubeStagedMatmulRoutineSettings::DEFAULT)
            .expect("the default prepares here");
        assert!(matches!(
            blueprint.microkernel.plane.instruction,
            PlaneFragmentInstruction::Cmma { .. } | PlaneFragmentInstruction::Mma { .. }
        ));
        assert_eq!(blueprint.microkernel.scratch, Scratch::OneTile);
    }

    /// Every width is the widest the device reads `f16` in, until an access's stretch or the
    /// binding's mask narrows it: the stages's stage and the fragments' tiles are sixteen deep, so
    /// only a mask does here.
    #[test]
    fn the_widths_are_what_the_accesses_and_the_binding_admit() {
        let hw = offering(devices::strix_halo());
        let widest = hw.vector_ceiling(F16);
        let widths = |m, n, k| {
            let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
            let settings = CubeStagedMatmulRoutineSettings::DEFAULT;
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the default prepares here");
            let widths = blueprint.widths;
            (widths.lhs, widths.rhs, widths.out)
        };
        assert_eq!(widths(256, 512, 64), (widest, widest, widest));
        // A `k` the stages overhang masks the lhs, whose contiguous axis it is.
        assert_eq!(widths(256, 512, 24), (1, widest, widest));
        // An `n` the cubes overhang masks the rhs and the output.
        assert_eq!(widths(256, 520, 64), (widest, 1, 1));
    }

    /// A padded stage holds its cells and one 16-byte chunk after every block row, a line of
    /// `f16` on this device being 16 bytes too: the default's stage, a `128 × 16` lhs and a
    /// `16 × 256` rhs, is 12 KB of cells and 128 + 256 rows of 16-byte blocks, whichever order the
    /// rhs binds in. In order or swizzled, the same stage holds its cells alone. The default's own,
    /// staged for `cmma` here, is the padded one.
    #[test]
    fn a_stage_counts_a_chunk_after_every_block_row_only_when_padded() {
        let hw = offering(devices::strix_halo());
        for problem in [problem(), col_weight()] {
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &CubeStagedMatmulRoutineSettings::DEFAULT)
                .expect("the default prepares");
            let StageStorage::Tiled { block, .. } = blueprint.shared_memory.clone() else {
                panic!("a matrix instruction stages in tiles");
            };
            let tiled = |chunks| StageStorage::Tiled {
                block: block.clone(),
                chunks,
            };
            let bytes = |storage: &StageStorage| {
                blueprint.partitioning.stage().bytes(
                    &hw,
                    problem.elements,
                    storage,
                    blueprint.microkernel.plane.orders,
                )
            };
            assert_eq!(bytes(&tiled(RowChunks::Padded)), 12288 + (128 + 256) * 16);
            assert_eq!(bytes(&tiled(RowChunks::InOrder)), 12288);
            assert_eq!(bytes(&tiled(RowChunks::Swizzled)), 12288);
            assert_eq!(
                bytes(&blueprint.shared_memory),
                bytes(&tiled(RowChunks::Padded))
            );
        }
    }

    /// The setting reaches the stage: rows in order resolve to an in-order stage under a matrix
    /// instruction, which holds its cells alone; on the software instruction, whose stage is
    /// strided either way, both layouts prepare one launch, so a table times it once.
    #[test]
    fn rows_in_order_lay_a_matrix_instructions_stage_and_leave_the_software_ones_alone() {
        let in_order = CubeStagedMatmulRoutineSettings {
            microkernel: CubeStagedMicrokernelSettings {
                stage_rows: CubeStagedRows::InOrder,
                ..CubeStagedMatmulRoutineSettings::DEFAULT.microkernel
            },
            ..CubeStagedMatmulRoutineSettings::DEFAULT
        };
        let prepared = |hw: &Hardware, problem: &MatmulRoutineProblem, settings| {
            CubeStagedMatmulRoutine
                .prepare(hw, problem, settings)
                .expect("the default's box prepares")
                .0
        };

        let hw = offering(devices::strix_halo());
        let blueprint = prepared(&hw, &problem(), &in_order);
        assert!(matches!(
            blueprint.shared_memory,
            StageStorage::Tiled {
                chunks: RowChunks::InOrder,
                ..
            }
        ));
        let bytes = blueprint.partitioning.stage().bytes(
            &hw,
            problem().elements,
            &blueprint.shared_memory,
            blueprint.microkernel.plane.orders,
        );
        assert_eq!(bytes, 12288);

        let software = devices::wgpu_no_subgroups();
        let f32_problem = MatmulRoutineProblem::dense(37, 29, 19, F32, MatmulLayout::Row);
        assert_eq!(
            prepared(&software, &f32_problem, &in_order),
            prepared(
                &software,
                &f32_problem,
                &CubeStagedMatmulRoutineSettings::DEFAULT
            )
        );
    }

    /// The planes meet in shared memory, so a device that cannot add an `f32` there declines,
    /// whatever instruction it offers.
    #[test]
    fn declines_without_an_atomic_add_in_shared_memory() {
        let mut hw = offering(devices::strix_halo());
        devices::features(&mut hw).types.shared_atomic.clear();
        let settings = CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K;
        assert_eq!(
            CubeStagedMatmulRoutine
                .prepare(&hw, &problem(), &settings)
                .err(),
            Some(Refusal::Declined {
                why: CubeStagedMicrokernel::DECLINE
            })
        );
    }

    /// A fragment cannot add into the accumulator through its own store, so a setting asking for
    /// no scratch runs with one fragment's window.
    #[test]
    fn a_fragment_always_bounces_into_the_accumulator() {
        let hw = offering(devices::strix_halo());
        let settings = split_with(|microkernel| microkernel.scratch = Scratch::None);
        let (blueprint, _) = CubeStagedMatmulRoutine
            .prepare(&hw, &problem(), &settings)
            .expect("the routine runs here");
        assert_eq!(blueprint.microkernel.scratch, Scratch::OneTile);
    }

    /// Planes running the software instruction add their cells into the accumulator on their
    /// own, and hold no window to bounce them through.
    #[test]
    fn the_software_instruction_adds_without_a_scratch() {
        let settings = CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K;
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&devices::strix_halo(), &problem(), &settings)
            .expect("the routine runs the software instruction here");
        let plane = blueprint.microkernel.plane;
        assert_eq!(
            plane.instruction,
            PlaneFragmentInstruction::Registers {
                step_sum: PlaneFragmentStepSum::Wide,
            }
        );
        assert_eq!(blueprint.microkernel.scratch, Scratch::None);
        assert_eq!(
            grid.planes().width,
            plane.instruction.units(&devices::strix_halo()).unwrap() as usize
        );
    }

    /// Planes split along `K` read their stages as wide as the device reads `f16`; the output is
    /// copied out of the atomics one element at a time.
    #[test]
    fn a_split_along_k_writes_its_output_one_element_at_a_time() {
        let hw = offering(devices::strix_halo());
        let widest = hw.vector_ceiling(F16);
        let widths = |m, n, k| {
            let problem = MatmulRoutineProblem::dense(m, n, k, F16, MatmulLayout::Row);
            let settings = CubeStagedMatmulRoutineSettings::SPLIT_ALONG_K;
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the default prepares here");
            let widths = blueprint.widths;
            (widths.lhs, widths.rhs, widths.out)
        };
        assert_eq!(widths(256, 512, 64), (widest, widest, 1));
        // A `k` the stages overhang masks the lhs, whose contiguous axis it is.
        assert_eq!(widths(256, 512, 72), (1, widest, 1));
    }

    /// Two NVFP4 inputs the block-scaled instruction reads as stored are estimated like any staged
    /// product. An estimate is what a table ranks a routine's candidates by, and one that took
    /// them for a weight its stages decode would ask for the scale blocks a stage of steps never
    /// holds.
    #[test]
    fn an_nvfp4_product_read_as_stored_is_estimated() {
        use crate::kernels::select::packed::nvfp4_scheme;
        use cubek_microkernel::matmul::dense::cube::CubeStagedDepth;
        use cubek_microkernel::matmul::stored::Operand;
        use cubek_microkernel::matmul::stored::Packing;
        use cubek_microkernel::matmul::tile_order;
        use cubek_std::microkernel::KernelEstimate;

        let hw = devices::offering_nvfp4(devices::h100());
        let packing = |scheme: cubecl::quant::scheme::QuantScheme, axes| Packing {
            operand: Operand::of_scheme(&scheme, K, axes).expect("NVFP4 is one block along k"),
            scheme,
        };
        let problem = MatmulRoutineProblem::packed(
            128,
            256,
            256,
            ElemType::Float(FloatKind::F16),
            MatmulOperands {
                lhs: Some(packing(nvfp4_scheme(), [M, K])),
                rhs: Some(packing(tile_order::scheme(), [K, N])),
                out: None,
            },
        );
        let (blueprint, grid) = CubeStagedMatmulRoutine
            .prepare(&hw, &problem, &CubeStagedMatmulRoutineSettings::DEFAULT)
            .expect("the default box prepares an NVFP4 product read as stored");
        assert!(matches!(
            blueprint.microkernel.depth,
            CubeStagedDepth::Steps
        ));
        let mut estimate = KernelEstimate::from(&grid);
        CubeStagedMatmulRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
        assert!(estimate.work.is_some());
    }

    /// A weight its stages decode is estimated a stage of scale blocks at a time: as many stages
    /// as `K` holds stages of the microkernel's blocks, split between the planes sharing `K`.
    #[test]
    fn a_decoded_weight_is_estimated_in_stages_of_its_blocks() {
        use cubek_microkernel::matmul::dense::cube::CubeStagedDepth;
        use cubek_microkernel::matmul::stored::Operand;
        use cubek_microkernel::matmul::stored::Packing;
        use cubek_microkernel::matmul::tile_order;
        use cubek_std::matmul_axes::KI;
        use cubek_std::microkernel::KernelEstimate;

        // The M2 as Metal reports it: its simdgroup matrix, `8 × 8 × 8` over `f16` into `f32`.
        let hw = devices::offering_cmma(
            devices::m2_pro(),
            [MmaConfig {
                a_type: ElemType::Float(FloatKind::F16),
                b_type: ElemType::Float(FloatKind::F16),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 8,
                n: 8,
                k: 8,
            }],
        );
        let scheme = tile_order::scheme();
        let weight = Packing {
            operand: Operand::of_scheme(&scheme, K, [K, N]).expect("NVFP4 is one block along k"),
            scheme,
        };
        let k = 1024;
        let problem = MatmulRoutineProblem::packed(
            128,
            256,
            k,
            ElemType::Float(FloatKind::F16),
            MatmulOperands {
                lhs: None,
                rhs: Some(weight),
                out: None,
            },
        );
        let mut estimated = 0;
        for settings in CubeStagedMatmulRoutine.packed_settings(&hw) {
            let Ok((blueprint, grid)) = CubeStagedMatmulRoutine.prepare(&hw, &problem, &settings)
            else {
                continue;
            };
            let CubeStagedDepth::ScaleBlocks(blocks) = blueprint.microkernel.depth else {
                panic!("{settings}: a decoded weight is staged in scale blocks");
            };
            let block = blueprint.partitioning.cube_box().extent(KI);
            let planes_along_k = blueprint.microkernel.planes_along_k.get();
            let mut estimate = KernelEstimate::from(&grid);
            CubeStagedMatmulRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
            let staging = estimate.staging.expect("a staged estimate");
            assert_eq!(
                staging.stages,
                k.div_ceil(block * blocks.get() * planes_along_k) as u64,
                "{settings}"
            );
            estimated += 1;
        }
        assert!(estimated > 0, "no packed setting prepared");
    }
}
