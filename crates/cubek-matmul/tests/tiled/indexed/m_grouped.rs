//! A product grouped along `M`: a mixture of experts' tokens sorted by expert, every group against
//! the slab its expert names. The rows are launched as chunks padded to whole tiles, each chunk
//! against its slab; a chunk of padding names one past the stack and answers zeros.

use core::marker::PhantomData;

use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::indexed::m_grouped::{MGroupedMatmulTensors, MGroupedSettings};
use cubek_matmul::kernels::indexed::stack::SlabAxis;
use cubek_matmul::kernels::tensor::StoredTensor;
use cubek_matmul::kernels::{
    CubeStagedMatmulRoutine, MatmulRoutine, MatmulRoutineProblem, PlaneShuffleMatmulRoutine,
};
use cubek_microkernel::matmul::{MatmulMicrokernel, MatmulOperands};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{
    HostData, HostDataType, ScopeUnderTest, TestOutcome, ValidationResult, Written,
};

use crate::matmul::Declines;
use crate::tiled::indexed::{Chunked, GroupedRows};
use crate::tiled::packed::{PackedFormat, PackedSlabs};
use crate::tiled::{AtSetting, ByRoutine, upload};

const FORMATS: [PackedFormat; 4] = [
    PackedFormat::Q8,
    PackedFormat::Q4,
    PackedFormat::Nvfp4,
    PackedFormat::Mxfp4,
];

#[test]
fn every_cube_staged_setting_reads_each_chunk_s_slab() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let chunked = Chunked {
        chunks: 5,
        rows: 32,
        slabs: 3,
        k: 96,
        n: 80,
    };
    staged.check_some(staged.at_every_setting(&[chunked]));
}

#[test]
fn every_plane_shuffle_setting_reads_each_chunk_s_slab() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let chunked = Chunked {
        chunks: 5,
        rows: 4,
        slabs: 3,
        k: 96,
        n: 80,
    };
    shuffle.check_some(shuffle.at_every_setting(&[chunked]));
}

/// Rows grouped against a stack packed in each format, through the plane shuffle's packed
/// settings.
#[test]
fn every_packed_plane_shuffle_setting_reads_each_group_s_packed_slab() {
    let grouped = ByGroupedRoutine::new(PlaneShuffleMatmulRoutine);
    grouped.check_some(grouped.at(
        MGroupedSettings::plane_shuffle_packed(),
        &FORMATS.map(GroupedRows::of),
    ));
}

/// Rows grouped against a stack packed in each format, through the cube staged routine's packed
/// settings, which stage a packed input for a matrix instruction: skipped on a device with none.
#[test]
fn every_packed_cube_staged_setting_reads_each_group_s_packed_slab() {
    let grouped = ByGroupedRoutine::new(CubeStagedMatmulRoutine);
    if grouped
        .hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .next()
        .is_none()
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the cube staged routine stages a packed input for a matrix instruction".to_string(),
        ))
        .enforce();
        return;
    }
    grouped.check_some(grouped.at(
        MGroupedSettings::cube_staged_packed(&grouped.hw),
        &FORMATS.map(GroupedRows::of),
    ));
}

/// The chunk rows of a cube staged setting are read from a prepare over every grouped row, and
/// the launch prepares again on the chunk rows alone: the cube tile it launches on is the rows the
/// groups were padded to, or the padded rows are read past their tiles.
#[test]
fn a_cube_staged_launch_holds_the_rows_its_groups_were_padded_to() {
    let (k, n, groups) = (64usize, 48usize, 4usize);
    let ends = [128u32, 128, 384, 512];
    let rows = *ends.last().unwrap() as usize;
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let mut walk = Walk::new();
    for settings in MGroupedSettings::cube_staged(&hw) {
        let tensors = MGroupedMatmulTensors::new(
            &client,
            upload(&client, &[rows, k], &vec![0f32; rows * k]),
            StoredTensor::Plain(upload(
                &client,
                &[k, groups * n],
                &vec![0f32; k * groups * n],
            )),
            upload(&client, &[groups], &ends),
            SlabAxis::Output,
        )
        .expect("a float stack derives a grouped launch");
        let Some(padded) = walk.ran(tensors.launch(&hw, &CubeStagedMatmulRoutine, &settings))
        else {
            continue;
        };
        let chunk = padded.layout().align();
        let problem = MatmulRoutineProblem::dense(chunk, n, k, ElemType::Float(FloatKind::F32));
        let Ok((blueprint, _)) = CubeStagedMatmulRoutine.prepare(&hw, &problem, &settings.routine)
        else {
            continue;
        };
        assert_eq!(blueprint.partitioning.rows() as usize, chunk, "{settings}");
    }
    walk.settle();
}

/// The same over a packed stack, in each format.
#[test]
fn a_cube_staged_launch_over_a_packed_stack_holds_the_rows_its_groups_were_padded_to() {
    let sizes = [100usize, 0, 300, 60];
    let rows: usize = sizes.iter().sum();
    let ends: Vec<u32> = sizes
        .iter()
        .scan(0, |end, size| {
            *end += size;
            Some(*end as u32)
        })
        .collect();
    let client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    if hw.properties.features.matmul.cmma.iter().next().is_none() {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the cube staged routine stages a packed input for a matrix instruction".to_string(),
        ))
        .enforce();
        return;
    }
    let mut walk = Walk::new();
    for format in FORMATS {
        let grouped = GroupedRows::of(format).sized(sizes);
        let stack = PackedSlabs::new(&client, grouped.packed_slabs());
        for settings in MGroupedSettings::cube_staged_packed(&hw) {
            let tensors = MGroupedMatmulTensors::new(
                &client,
                grouped.activation(&client),
                StoredTensor::Packed(stack.matrix()),
                upload(&client, &[sizes.len()], &ends),
                SlabAxis::Output,
            )
            .expect("a packed stack of whole slabs derives a grouped launch");
            let launched = tensors.launch(&hw, &CubeStagedMatmulRoutine, &settings);
            let Some(padded) = walk.ran(launched) else {
                continue;
            };
            let chunk = padded.layout().align();
            let packing = stack
                .matrix()
                .stacked(sizes.len(), SlabAxis::Output.axis())
                .expect("whole slabs stack")
                .packing();
            let problem = MatmulRoutineProblem::packed(
                chunk,
                grouped.n,
                grouped.k,
                ElemType::Float(FloatKind::F16),
                MatmulOperands {
                    rhs: Some(packing),
                    ..MatmulOperands::all(None)
                },
            );
            let (blueprint, _) = CubeStagedMatmulRoutine
                .prepare(&hw, &problem, &settings.routine)
                .expect("the launch prepared these chunk rows");
            assert_eq!(
                blueprint.partitioning.rows() as usize,
                chunk,
                "{format:?} {settings}"
            );
        }
    }
    walk.settle();
}

/// A routine reached through the grouped rows' own launch, at one of the grouped settings it
/// lists, against the host's product.
struct ByGroupedRoutine<R, M, SM> {
    routine: R,
    hw: Hardware,
    kinds: PhantomData<(M, SM)>,
}

impl<R, M, SM> ByGroupedRoutine<R, M, SM>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
{
    fn new(routine: R) -> Self {
        Self {
            routine,
            hw: Hardware::from_client(&cubecl::test_device().client()),
            kinds: PhantomData,
        }
    }

    /// Each of `settings` on each of `products`.
    fn at(
        &self,
        settings: Vec<MGroupedSettings<R::Settings>>,
        products: &[GroupedRows],
    ) -> Vec<AtSetting<MGroupedSettings<R::Settings>, GroupedRows>> {
        AtSetting::every(settings, products)
    }
}

impl<R, M, SM> ScopeUnderTest for ByGroupedRoutine<R, M, SM>
where
    R: MatmulRoutine<M, SM>,
    M: MatmulMicrokernel,
{
    type Case = AtSetting<MGroupedSettings<R::Settings>, GroupedRows>;

    fn launch(&self, client: &Client, case: &Self::Case) -> Result<Vec<Written>, Refusal> {
        let grouped = &case.product;
        let stack = PackedSlabs::new(client, grouped.packed_slabs());
        let tensors = MGroupedMatmulTensors::new(
            client,
            grouped.activation(client),
            StoredTensor::Packed(stack.matrix()),
            upload(client, &[grouped.sizes.len()], &grouped.ends()),
            SlabAxis::Output,
        )
        .expect("a packed stack of whole slabs derives a grouped launch");
        let padded = tensors.launch(&self.hw, &self.routine, &case.settings)?;
        Ok(vec![Written {
            name: "out",
            tensor: padded.unpad(),
            host: HostDataType::F32,
        }])
    }

    fn reference(&self, case: &Self::Case) -> Vec<HostData> {
        vec![case.product.reference()]
    }

    fn tolerance(&self, case: &Self::Case) -> f32 {
        case.product.tolerance()
    }
}

/// The launches of a walk over settings. At least one must run, unless every refusal was a cube
/// larger than the device holds (the CPU runtime's few units), which skips the test.
struct Walk {
    ran: usize,
    cubes_past_device: bool,
}

impl Walk {
    fn new() -> Self {
        Self {
            ran: 0,
            cubes_past_device: true,
        }
    }

    /// The launch's output when it ran, counting it; else `None`, noting why it did not.
    fn ran<T>(&mut self, launched: Result<T, Refusal>) -> Option<T> {
        match launched {
            Ok(out) => {
                self.ran += 1;
                Some(out)
            }
            Err(why) => {
                self.cubes_past_device &= matches!(why, Refusal::CubePastDevice { .. });
                None
            }
        }
    }

    fn settle(self) {
        if self.ran == 0 && self.cubes_past_device {
            TestOutcome::Validated(ValidationResult::Skipped(
                "every setting asks for a cube larger than this device holds".to_string(),
            ))
            .enforce();
            return;
        }
        assert!(self.ran > 0, "no setting prepared");
    }
}
