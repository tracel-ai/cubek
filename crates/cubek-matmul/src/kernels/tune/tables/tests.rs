use super::base::MatmulTableGroups;
use super::{block_scaled, few_rows, many_rows};
use cubek_std::{Hardware, devices};

/// Every table on every device holds the candidates `tables.txt` records: as many, under the same
/// names, which is what a persisted winner is checksummed by.
///
/// A line that moves is a table that races a candidate it did not, or stops racing one it did, and
/// every cached measurement of it is taken again: a change to a table rewrites the file on
/// purpose. Rewrite it with `CUBEK_WRITE_PLANS=1` and review the diff.
#[test]
fn every_table_holds_the_candidates_written_down() {
    let written = render_all();
    if std::env::var_os("CUBEK_WRITE_PLANS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/kernels/tune/tables/tables.txt"
        );
        std::fs::write(path, &written).expect("tables.txt is writable");
        return;
    }
    assert_eq!(
        written,
        include_str!("tables.txt"),
        "a table changed; rewrite tables.txt with CUBEK_WRITE_PLANS=1 and review the diff"
    );
}

/// Every table on every device, one line each: its candidate count and its names' checksum.
fn render_all() -> String {
    let groups = MatmulTableGroups::default();
    let mut out = String::new();
    for (device, hw) in devices_listed() {
        out.push_str(&format!("# {device}\n"));
        for (table, built) in [
            ("few-rows", few_rows::table(hw.clone(), groups)),
            ("many-rows", many_rows::table(hw.clone(), groups)),
            ("block-scaled", block_scaled::table(hw.clone(), groups)),
        ] {
            out.push_str(&format!(
                "{table}: {} candidates, checksum {}\n",
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
