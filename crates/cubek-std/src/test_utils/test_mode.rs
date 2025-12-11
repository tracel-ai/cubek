const CUBEK_TEST_MODE_ENV: &str = "CUBEK_TEST_MODE";

#[derive(Default)]
pub enum TestMode {
    #[default]
    /// Tests resulting in compilation error are marked as `ok`
    Skip,
    /// Tests resulting in compilation error are marked as `failed`
    Panic,
    /// Tests are marked as `failed` and all data is shown
    Print,
}

pub fn current_test_mode() -> TestMode {
    let env = std::env::var(CUBEK_TEST_MODE_ENV);

    match env {
        Ok(val) => match val.to_lowercase().as_str() {
            "skip" => TestMode::Skip,
            "panic" => TestMode::Panic,
            "print" => TestMode::Print,
            _ => TestMode::default(),
        },
        Err(_) => TestMode::default(),
    }
}
