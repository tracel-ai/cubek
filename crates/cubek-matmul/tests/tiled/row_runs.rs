//! Rows in runs of a flat, unpadded activation against a packed stack: each run reads and writes
//! only its own rows, so the poison the output starts as stays where no run reaches.

use core::num::NonZeroUsize;
use cubecl::{client::Client, std::tensor::TensorHandle, tile::launch::Decline, zspace::Shape};
use cubek_matmul::kernels::PlaneShuffleMatmulRoutine;
use cubek_matmul::kernels::{MatmulIndex, MatmulIndexTable, MatmulRoutine, RowEnds};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperand};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataVec, ScopeChecker, TestOutcome, ValidationResult};
use half::f16;

use crate::matmul::Declines;
use crate::tiled::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use crate::tiled::{ByRoutine, RoutineProduct, upload};

const SLABS: usize = 4;
const N: usize = 96;
/// The most rows a run holds.
const ROWS_PER_RUN: usize = 4;
/// Each run's `(rows, slab)`: an empty first one, one ending mid-tile, an empty middle one, a full
/// one, one on the slab past the stack and an empty sentinel; seven more rows no run covers.
const RUNS: [(usize, usize); 6] = [(0, 0), (3, 1), (0, 2), (4, 2), (2, SLABS), (0, SLABS)];
const ROWS: usize = 9 + 7;

/// The runs over a stack of NVFP4 slabs `k` deep.
#[derive(Debug, Clone)]
struct RowRuns {
    k: usize,
}

impl RowRuns {
    /// Each run's `(start, end, slab)`.
    fn runs(&self) -> Vec<(usize, usize, u32)> {
        let ends = RUNS.iter().scan(0, |end, &(rows, _)| {
            *end += rows;
            Some(*end)
        });
        let runs = ends.zip(RUNS);
        runs.map(|(end, (rows, slab))| (end - rows, end, slab as u32))
            .collect()
    }

    fn stack(&self) -> PackedSlabsSpec {
        PackedSlabsSpec {
            format: PackedFormat::Nvfp4,
            operand: MatmulOperand::Rhs,
            slabs: SLABS,
            rows: N,
            k: self.k,
            seed: 2,
        }
    }

    fn activation(&self) -> Vec<f16> {
        (0..ROWS * self.k)
            .map(|i| f16::from_f32(((i * 7919 + 104_729) % 2003) as f32 / 1001.5 - 1.))
            .collect()
    }
}

impl RoutineProduct for RowRuns {
    fn launch<R, M, SM>(
        &self,
        routine: &R,
        hw: &Hardware,
        client: &Client,
        settings: &R::Settings,
    ) -> Result<TensorHandle, Decline>
    where
        R: MatmulRoutine<M, SM>,
        M: MatmulMicrokernel,
    {
        let lhs = upload(client, &[ROWS, self.k], &self.activation());
        let rhs = PackedSlabs::new(client, self.stack()).stack();
        let runs = self.runs();
        let column = |at: fn(&(usize, usize, u32)) -> u32| {
            upload(
                client,
                &[runs.len()],
                &runs.iter().map(at).collect::<Vec<_>>(),
            )
        };
        let table = MatmulIndexTable {
            index: MatmulIndex::RunOfMOnSlab { slabs: SLABS },
            entries: column(|run| run.2),
            row_ends: Some(RowEnds {
                ends: column(|run| run.1 as u32),
                rows_per_run: NonZeroUsize::new(ROWS_PER_RUN).expect("runs hold rows"),
            }),
        };
        let (out, bound) = Self::poisoned(client, Shape::from(vec![ROWS, N]), lhs.elem);
        routine.launch_indexed_into(hw, client, lhs, rhs, table, bound, settings)?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let lhs: Vec<f32> = self.activation().iter().map(|half| half.to_f32()).collect();
        let stack = self.stack().dequantized();
        let mut out = vec![f32::NAN; ROWS * N];
        for (start, end, slab) in self.runs() {
            for (row, j) in (start..end).flat_map(|row| (0..N).map(move |j| (row, j))) {
                let past_stack = slab as usize >= SLABS;
                out[row * N + j] = (0..self.k)
                    .map(|k| match past_stack {
                        true => 0.,
                        false => lhs[row * self.k + k] * stack.cell(slab as usize, k, j),
                    })
                    .sum();
            }
        }
        HostData::row_major(HostDataVec::F32(out), Shape::from(vec![ROWS, N]))
    }

    fn tolerance(&self) -> f32 {
        1e-2
    }
}

/// Every plane shuffle setting that serves the product writes the rows of each run and nothing
/// else; a skip needs every setting refused for what the device lacks.
#[test]
fn plane_shuffle_settings_write_only_the_rows_of_each_run() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let products = [128, 160].map(|k| RowRuns { k });
    let settings = PlaneShuffleMatmulRoutine::indexed_packed_settings();
    let client = cubecl::test_device().client();
    let (mut launched, mut refused) = (0, Vec::new());
    for case in shuffle.at(settings, &products) {
        match shuffle.attempt(&client, &case) {
            Ok(outcome) => {
                launched += 1;
                outcome.enforce();
            }
            Err(why) => refused.push(why),
        }
    }
    let by_device = |why: &Decline| {
        let past = matches!(
            why,
            Decline::CubePastDevice { .. } | Decline::CubeShapePastDevice { .. }
        );
        past || matches!(
            why,
            Decline::GridPastDevice { .. } | Decline::SharedMemoryPastDevice { .. }
        ) || matches!(
            why,
            Decline::PlanesUnavailable | Decline::ImprecisePlaneDim { .. }
        )
    };
    if launched == 0 {
        assert!(refused.iter().all(by_device), "refused: {refused:?}");
        let skipped = ValidationResult::Skipped("the device refuses every setting".into());
        TestOutcome::Validated(skipped).enforce();
    }
}
