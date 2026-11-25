//! Every routine's estimate is whole: each part set by its owner, and the cubes performing at
//! least what the problem asks.

/// The routines a product of many rows races: their estimates on a strix halo.
mod many_rows_estimates {
    use crate::kernels::CubeStagedMatmulRoutine;
    use crate::kernels::UnitRegistersMatmulRoutine;
    use crate::kernels::{MatmulRoutine, MatmulRoutineProblem};
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::MatmulMicrokernel;
    use cubek_std::Hardware;
    use cubek_std::devices;
    use cubek_std::microkernel::KernelEstimate;
    use cubek_std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// The dev machine's Radeon 8060S offering the `16×16×16` `f16` instruction summed in
    /// `f32`.
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

    /// Every part of `R`'s estimate at every setting it lists, on an aligned product and
    /// a ragged one: each owner set its own, and the cubes perform at least what the problem
    /// asks.
    fn assert_every_estimate_is_whole<M: MatmulMicrokernel, SM, R: MatmulRoutine<M, SM>>() {
        let hw = &strix_halo();
        let mut prepared = 0;
        for settings in R::settings(hw) {
            for (m, n, k) in [(2048, 2048, 6144), (37, 29, 19)] {
                let problem = MatmulRoutineProblem::dense(m, n, k, F16);
                let Ok((blueprint, grid)) = R::prepare(hw, &problem, &settings) else {
                    continue;
                };
                prepared += 1;
                let mut estimate = KernelEstimate::from(&grid);
                R::estimate(&problem, &blueprint, &grid, &mut estimate);
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
        assert_every_estimate_is_whole::<_, _, UnitRegistersMatmulRoutine>();
        assert_every_estimate_is_whole::<_, _, CubeStagedMatmulRoutine>();
        assert_every_estimate_is_whole::<_, _, crate::kernels::HypercubeLastArrivalMatmulRoutine>();
    }
}

/// The routines a product of few rows races: their estimates on an rtx2060.
mod few_rows_estimates {
    use crate::kernels::CubeShuffleMatmulRoutine;
    use crate::kernels::PlaneShuffleMatmulRoutine;
    use crate::kernels::{MatmulRoutine, MatmulRoutineProblem};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::matmul::{MatmulLayout, MatmulMicrokernel, MatmulOperand};
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

    /// Every part of `R`'s estimate at every setting it lists, on decode projections and
    /// a ragged product: each owner set its own, and the cubes perform at least what the
    /// problem asks.
    fn assert_every_estimate_is_whole<Mk: MatmulMicrokernel, R: MatmulRoutine<Mk>>() {
        let hw = &rtx2060();
        let mut prepared = 0;
        for settings in R::settings(hw) {
            for (m, n, k) in [(1, 2048, 6144), (8, 4096, 2048), (3, 29, 1024)] {
                for rhs_layout in [MatmulLayout::Col, MatmulLayout::Row] {
                    let problem = MatmulRoutineProblem::dense(m, n, k, F16)
                        .laid(MatmulOperand::Rhs, rhs_layout);
                    let Ok((blueprint, grid)) = R::prepare(hw, &problem, &settings) else {
                        continue;
                    };
                    prepared += 1;
                    let mut estimate = KernelEstimate::from(&grid);
                    R::estimate(&problem, &blueprint, &grid, &mut estimate);
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
        assert_every_estimate_is_whole::<_, PlaneShuffleMatmulRoutine>();
        assert_every_estimate_is_whole::<_, CubeShuffleMatmulRoutine>();
    }
}
