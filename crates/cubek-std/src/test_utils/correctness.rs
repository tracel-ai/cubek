use crate::test_utils::test_mode::{TestMode, current_test_mode};
use crate::test_utils::test_tensor::new_casted;
use cubecl::CubeElement;
use cubecl::frontend::CubePrimitive;
use cubecl::{TestRuntime, client::ComputeClient, std::tensor::TensorHandle};

/// Compares the content of a handle to a given slice of f32.
pub fn assert_equals_approx(
    client: &ComputeClient<TestRuntime>,
    out: &TensorHandle<TestRuntime>,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    // Obtain the data in f32 for not being generic over type
    let data_handle = new_casted(client, out, f32::as_type_native_unchecked());
    let data_f32 =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    if matches!(current_test_mode(), TestMode::Print) {
        println!("Epsilon: {:?}", epsilon);
    }

    for (i, (a, e)) in data_f32.iter().zip(expected.iter()).enumerate() {
        let allowed_error = (epsilon * e).max(epsilon);

        if matches!(current_test_mode(), TestMode::Print) {
            println!("{:?}: {:?}, {:?}", i, a, e);
        } else {
            let actual_nan = f32::is_nan(*a);
            let expected_nan = f32::is_nan(*e);

            if actual_nan != expected_nan {
                if expected_nan {
                    return Err(format!("Expected NaN, got value={:?}", *a));
                } else {
                    return Err(format!("Expected value={:?}, got NaN", *e));
                }
            }

            let difference = f32::abs(a - e);

            if difference >= allowed_error {
                return Err(format!(
                    "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                    i, *a, *e, difference, epsilon
                ));
            }
        }
    }

    if matches!(current_test_mode(), TestMode::Print) {
        Err("".to_string())
    } else {
        Ok(())
    }
}
