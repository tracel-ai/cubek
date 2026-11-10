//! The cube groups routine, the autotune's floor: a row a group of a cube's units.

use cubecl::zspace::shape;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_rms_norm::kernels::{CubeGroupsRmsNormRoutine, RmsNormCall, RmsNormRoutineCandidate};
use cubek_std::Hardware;
use cubek_test_utils::{ScopeUnderTest, TestInput};

use crate::rms_norm::{Declines, EPSILON, F32, RmsNorm};

/// The floor serves every case at every setting: a decline would be a family with nothing to fall
/// back on.
#[test]
fn cube_groups_routine_matches_the_reference() {
    let rms_norm = RmsNorm::new(CubeGroupsRmsNormRoutine, Declines::Never);
    rms_norm.check(rms_norm.every_case());
}

/// No rows is no launch: the floor hands back an empty norm.
#[test]
fn no_rows_launch_nothing() {
    let client = cubecl::test_device().client();
    let candidate = RmsNormRoutineCandidate::floor(&Hardware::from_client(&client));
    let tensor = |values: Vec<f32>, shape| {
        PlainTensor::of_handle(
            TestInput::builder(client.clone(), shape)
                .dtype(F32)
                .custom(values)
                .generate_without_host_data(),
        )
    };
    let call = RmsNormCall {
        client: client.clone(),
        input: tensor(Vec::new(), shape![0, 128]),
        weight: tensor(vec![1.0; 128], shape![128]),
        epsilon: EPSILON,
    };
    let out = candidate
        .launch(&call)
        .unwrap_or_else(|why| panic!("the floor refused no rows: {why:?}"));
    assert_eq!(out.shape().num_elements(), 0);
}
