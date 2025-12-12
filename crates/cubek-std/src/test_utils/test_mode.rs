const CUBEK_TEST_MODE_ENV: &str = "CUBEK_TEST_MODE";

#[derive(Default)]
pub enum TestMode {
    #[default]
    /// Tests resulting in compilation error are marked as `ok`
    Skip,
    /// Tests resulting in compilation error are marked as `failed`
    Panic,
    /// Tests are marked as `failed` and all data is shown up to max data
    Print { max: usize },
}

pub fn current_test_mode() -> TestMode {
    match std::env::var(CUBEK_TEST_MODE_ENV) {
        Ok(val) => {
            let val = val.to_lowercase();
            if val.starts_with("print") {
                // Try to parse `print:42`
                if let Some((_, n)) = val.split_once(':') {
                    if let Ok(max) = n.parse() {
                        return TestMode::Print { max };
                    }
                }
                // Fallback if no number given
                TestMode::Print { max: 10 }
            } else if val == "panic" {
                TestMode::Panic
            } else {
                TestMode::Skip
            }
        }
        Err(_) => TestMode::default(),
    }
}
