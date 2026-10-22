#![cfg(all(feature = "std", not(target_family = "wasm")))]

use cubek_std::config::{CubekConfig, MatmulConfig, RuntimeConfig};

// Separate processes isolate the immutable configuration and working directory.
#[test]
fn configuration_sources() {
    for (file, content, expected) in [
        ("cubek.toml", "", true),
        (
            "cubek.toml",
            "[matmul]\nallow_tf32 = false\n[test]\npolicy = \"strict\"\n",
            false,
        ),
        ("burn.toml", "[cubek.matmul]\nallow_tf32 = false\n", false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(file), content).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "configuration_child", "--nocapture"])
            .env("CUBEK_CONFIG_TEST_EXPECTED", expected.to_string())
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{file}: {content}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn configuration_child() {
    let Ok(expected) = std::env::var("CUBEK_CONFIG_TEST_EXPECTED") else {
        return;
    };
    let expected = expected.parse::<bool>().unwrap();
    assert_eq!(MatmulConfig::get().allow_tf32, expected);
    assert_eq!(CubekConfig::get().matmul.allow_tf32, expected);
}
