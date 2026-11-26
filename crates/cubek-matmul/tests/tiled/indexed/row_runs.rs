//! Rows cut to runs of a flat, unpadded activation: each tile reads the rows its run covers, against
//! the slab its entry names, and writes the same rows of a flat output. Rows past a run are neither
//! read nor written, so the poison the output starts as stays where no tile reaches.

use cubek_matmul::kernels::indexed::m_grouped::MGroupedSettings;
use cubek_matmul::kernels::{CubeStagedMatmulRoutine, PlaneShuffleMatmulRoutine};
use cubek_test_utils::{ScopeUnderTest, TestOutcome, ValidationResult};

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::indexed::RowRuns;
use crate::tiled::packed::PackedFormat;

/// Through the cube staged routine's packed settings, which stage a packed input for a matrix
/// instruction: skipped on a device with none.
#[test]
fn every_packed_cube_staged_setting_reads_and_writes_the_rows_of_each_run() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    if staged
        .hw()
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
    let settings: Vec<_> = MGroupedSettings::cube_staged_packed(staged.hw())
        .into_iter()
        .map(|settings| settings.routine)
        .collect();
    let products = [RowRuns::ragged(32, 128, PackedFormat::Nvfp4)];
    staged.check_some(staged.at(settings, &products));
}

/// Through the plane shuffle's packed settings, those reading a scale block at a time included,
/// over a `k` of whole words of scales and one of two and a half.
#[test]
fn every_packed_plane_shuffle_setting_reads_and_writes_the_rows_of_each_run() {
    let shuffle = ByRoutine::new(PlaneShuffleMatmulRoutine, Declines::AsItStates);
    let products = [128, 160].map(|k| RowRuns::ragged(4, k, PackedFormat::Nvfp4));
    shuffle.check_some(shuffle.at(
        PlaneShuffleMatmulRoutine::indexed_packed_settings(),
        &products,
    ));
}
