use crate::test_utils::correctness::{TensorFilter, parse_tensor_filter};

const CUBEK_TEST_MODE_ENV: &str = "CUBEK_TEST_MODE";

#[derive(Default, Debug)]
pub enum TestMode {
    #[default]
    /// Tests resulting in compilation error are marked as `ok`
    Skip,
    /// Tests resulting in compilation error are marked as `failed`
    Panic,
    /// Tests are marked as `failed` and all data is shown up to max data
    Print(TensorFilter),
}

pub fn current_test_mode() -> TestMode {
    match std::env::var(CUBEK_TEST_MODE_ENV) {
        Ok(val) => {
            let val = val.to_lowercase();
            if val.starts_with("print") {
                if let Some((_, f)) = val.split_once(':') {
                    match parse_tensor_filter(f) {
                        Ok(filter) => TestMode::Print(filter),
                        Err(e) => {
                            eprintln!("Invalid print filter '{}': {}", f, e);
                            TestMode::Print(vec![]) // fallback wildcard
                        }
                    }
                } else {
                    TestMode::Print(vec![]) // wildcard
                }
            } else if val == "panic" {
                TestMode::Panic
            } else {
                TestMode::Skip
            }
        }
        Err(_) => TestMode::Skip,
    }
}
