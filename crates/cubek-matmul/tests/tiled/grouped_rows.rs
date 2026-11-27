//! Row groups of a flat, unpadded activation against a packed stack: each row group reads and writes
//! only its own rows, so the poison the output starts as stays where no row group reaches.

use core::num::NonZeroUsize;
use cubecl::{client::Client, std::tensor::TensorHandle, tile::launch::Decline, zspace::Shape};
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulIndexTable, MatmulRoutine,
    PlaneShuffleMatmulRoutine, RowGroups,
};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperand};
use cubek_std::Hardware;
use cubek_test_utils::{HostData, HostDataVec, ScopeChecker, TestOutcome, ValidationResult};
use half::f16;

use crate::matmul::Declines;
use crate::tiled::packed::{PackedFormat, PackedSlabs, PackedSlabsSpec};
use crate::tiled::{ByRoutine, RoutineProduct, upload};

const SLABS: usize = 4;
const N: usize = 96;
/// Rows past the last row group, which none covers.
const UNCOVERED_ROWS: usize = 7;

/// The row groups over a stack of NVFP4 slabs `k` deep, none longer than `max_rows`.
#[derive(Debug, Clone)]
struct GroupedRows {
    k: usize,
    max_rows: usize,
}

impl GroupedRows {
    /// Each row group's `(rows, slab)`: an empty first one, one ending mid-tile, an empty middle
    /// one, a full one, one on the slab past the stack and an empty sentinel.
    fn sizes(&self) -> [(usize, usize); 6] {
        let max = self.max_rows;
        [
            (0, 0),
            (max - 1, 1),
            (0, 2),
            (max, 2),
            (2, SLABS),
            (0, SLABS),
        ]
    }

    /// Each row group's `(start, end, slab)`.
    fn groups(&self) -> Vec<(usize, usize, u32)> {
        let sizes = self.sizes();
        let ends = sizes.iter().scan(0, |end, &(rows, _)| {
            *end += rows;
            Some(*end)
        });
        ends.zip(sizes)
            .map(|(end, (rows, slab))| (end - rows, end, slab as u32))
            .collect()
    }

    fn rows(&self) -> usize {
        self.sizes().iter().map(|(rows, _)| rows).sum::<usize>() + UNCOVERED_ROWS
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
        (0..self.rows() * self.k)
            .map(|i| f16::from_f32(((i * 7919 + 104_729) % 2003) as f32 / 1001.5 - 1.))
            .collect()
    }
}

impl RoutineProduct for GroupedRows {
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
        let rows = self.rows();
        let lhs = upload(client, &[rows, self.k], &self.activation());
        let rhs = PackedSlabs::new(client, self.stack()).stack();
        let groups = self.groups();
        let column = |at: fn(&(usize, usize, u32)) -> u32| {
            upload(
                client,
                &[groups.len()],
                &groups.iter().map(at).collect::<Vec<_>>(),
            )
        };
        let table = MatmulIndexTable::GroupedRows {
            slabs: SLABS,
            entries: column(|group| group.2),
            groups: RowGroups {
                ends: column(|group| group.1 as u32),
                max_rows: NonZeroUsize::new(self.max_rows).expect("row groups hold rows"),
            },
        };
        let (out, bound) = Self::poisoned(client, Shape::from(vec![rows, N]), lhs.elem);
        routine.launch_indexed_into(hw, client, lhs, rhs, table, bound, settings)?;
        Ok(out)
    }

    fn reference(&self) -> HostData {
        let lhs: Vec<f32> = self.activation().iter().map(|half| half.to_f32()).collect();
        let stack = self.stack().dequantized();
        let rows = self.rows();
        let mut out = vec![f32::NAN; rows * N];
        for (start, end, slab) in self.groups() {
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
        HostData::row_major(HostDataVec::F32(out), Shape::from(vec![rows, N]))
    }

    fn tolerance(&self) -> f32 {
        1e-2
    }
}

/// Every plane shuffle setting that serves the product writes the rows of each row group and nothing
/// else; a skip needs every setting refused for what the device lacks.
#[test]
fn plane_shuffle_settings_write_only_the_rows_of_each_row_group() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let products = [128, 160].map(|k| GroupedRows { k, max_rows: 4 });
    let settings = PlaneShuffleMatmulRoutine::indexed_packed_settings();
    let client = cubecl::test_device().client();
    let (mut launched, mut refused) = (0, Vec::new());
    for case in shuffle.at(settings, &products) {
        match shuffle.attempt(&client, &case) {
            Ok(outcome) => launched += enforced(outcome),
            Err(why) => refused.push(why),
        }
    }
    skip_unless_launched(launched, &refused);
}

/// A cube holds a row group in one box, so only the products whose `max_rows` is as tall as the
/// box launch: at least one must, and each that does writes only the rows of each row group.
#[test]
fn cube_staged_writes_only_the_rows_of_each_row_group_a_box_holds() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let products = [16, 32, 64, 128].map(|max_rows| GroupedRows { k: 128, max_rows });
    let settings = [CubeStagedMatmulRoutineSettings::PACKED];
    let client = cubecl::test_device().client();
    let (mut launched, mut refused) = (0, Vec::new());
    for case in staged.at(settings, &products) {
        match staged.attempt(&client, &case) {
            Ok(outcome) => launched += enforced(outcome),
            Err(why) => refused.push(why),
        }
    }
    skip_unless_launched(launched, &refused);
}

/// One for a case that launched, its outcome enforced.
fn enforced(outcome: TestOutcome) -> usize {
    outcome.enforce();
    1
}

/// Where no case launched, a skip, provided every refusal is for what the device lacks.
fn skip_unless_launched(launched: usize, refused: &[Decline]) {
    if launched == 0 {
        skip_refused_by_device(refused)
    }
}

fn skip_refused_by_device(refused: &[Decline]) {
    assert!(
        refused.iter().all(refused_by_device),
        "refused: {refused:?}"
    );
    let skipped = ValidationResult::Skipped("the device refuses every setting".into());
    TestOutcome::Validated(skipped).enforce();
}

/// Whether `why` names something the device lacks rather than something the product breaks.
fn refused_by_device(why: &Decline) -> bool {
    matches!(
        why,
        Decline::CubePastDevice { .. }
            | Decline::CubeShapePastDevice { .. }
            | Decline::GridPastDevice { .. }
            | Decline::SharedMemoryPastDevice { .. }
            | Decline::PlanesUnavailable
            | Decline::ImprecisePlaneDim { .. }
    )
}
