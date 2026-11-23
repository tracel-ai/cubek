use super::MatmulAutotuneGroups;
use super::{block_scaled, few_rows, many_rows};
use cubek_std::{Hardware, devices};

/// Every autotune on every device holds the candidates `autotunes.txt` records: as many, under the
/// same names, which is what a persisted winner is checksummed by.
///
/// A line that moves is an autotune that races a candidate it did not, or stops racing one it did,
/// and every cached measurement of it is taken again: a change to an autotune rewrites the file on
/// purpose. Rewrite it with `CUBEK_WRITE_PLANS=1` and review the diff.
#[test]
fn every_table_holds_the_candidates_written_down() {
    let written = render_all();
    if std::env::var_os("CUBEK_WRITE_PLANS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/kernels/tune/autotunes/autotunes.txt"
        );
        std::fs::write(path, &written).expect("autotunes.txt is writable");
        return;
    }
    assert_eq!(
        written,
        include_str!("autotunes.txt"),
        "an autotune changed; rewrite autotunes.txt with CUBEK_WRITE_PLANS=1 and review the diff"
    );
}

/// Every autotune on every device, one line each: its candidate count and its names' checksum.
fn render_all() -> String {
    let groups = MatmulAutotuneGroups::default();
    let mut out = String::new();
    for (device, hw) in devices_listed() {
        out.push_str(&format!("# {device}\n"));
        for (autotune, built) in [
            ("few-rows", few_rows::candidates(hw.clone(), groups)),
            ("many-rows", many_rows::candidates(hw.clone(), groups)),
            ("block-scaled", block_scaled::candidates(hw.clone(), groups)),
        ] {
            out.push_str(&format!(
                "{autotune}: {} candidates, checksum {}\n",
                built.len(),
                built.compute_checksum()
            ));
        }
        out.push('\n');
    }
    out
}

/// The devices whose settings lists differ: a Mac, a datacenter GPU with and without the 4-bit
/// instruction, and a device without planes.
fn devices_listed() -> [(&'static str, Hardware); 4] {
    [
        ("m2-pro", devices::m2_pro()),
        ("h100", devices::h100()),
        ("h100 with nvfp4", devices::offering_nvfp4(devices::h100())),
        ("wgpu without subgroups", devices::wgpu_no_subgroups()),
    ]
}

/// **A race runs on the call's buffers.** The few-rows autotune hands its candidates the call's own
/// lhs and rhs, and a destination the call allocated once at the product's shape, which the winner
/// then writes too.
#[test]
fn a_race_runs_on_the_calls_buffers() {
    use crate::kernels::plan::{MatmulCall, MatmulCallInput};
    use crate::kernels::tensor::{PlainTensor, StoredTensor};
    use cubecl::CubeTuneId;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    let client = cubek_test_utils::client();
    let input = |shape: [usize; 2]| {
        MatmulCallInput::plain(PlainTensor::zeros(&client, &shape, [shape[1], 1], F32))
    };
    let call = MatmulCall {
        hw: Hardware::from_client(&client),
        client: client.clone(),
        tune_id: CubeTuneId::new(&client, &cubecl::test_device()),
        lhs: input([3, 5]),
        rhs: input([5, 4]),
        out: F32,
        destination: None,
    }
    .with_destination();
    let destination = call.destination.clone().expect("a destination allocated");
    assert_eq!(destination.shape().as_slice(), [3, 4]);

    let tunables = few_rows::candidates(call.hw.clone(), MatmulAutotuneGroups::default());
    let raced = tunables.generate_inputs(&call.key(), &call);
    let memory = |input: &MatmulCallInput| match &input.tensor {
        StoredTensor::Plain(tensor) => tensor.binding.handle.memory.id(),
        StoredTensor::Packed(_) => unreachable!("a float call"),
    };
    assert_eq!(memory(&raced.lhs), memory(&call.lhs));
    assert_eq!(memory(&raced.rhs), memory(&call.rhs));
    let raced_destination = raced.destination.expect("the call's destination");
    assert!(matches!(
        raced_destination.handle.overlap(&destination.handle),
        cubecl::server::Overlap::Same
    ));
}
