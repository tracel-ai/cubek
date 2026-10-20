#![cfg(all(feature = "std", not(target_family = "wasm")))]

use cubek_std::config::{CubekConfig, MatmulConfig, RuntimeConfig};

// Each case gets its own process: changing an initialized config is intentionally
// unsupported, and changing cwd/environment in a parallel test suite is unsafe.
#[test]
fn configuration_sources() {
    for case in [
        "defaults",
        "standalone",
        "burn",
        "standalone_precedence",
        "set_precedence",
        "concurrent_read",
    ] {
        let dir = tempfile::tempdir().unwrap();
        // Stop the search here even when testing defaults.
        std::fs::write(dir.path().join("burn.toml"), "[cubek]\n").unwrap();
        match case {
            "standalone" | "concurrent_read" => {
                std::fs::write(
                    dir.path().join("cubek.toml"),
                    "[matmul]\nallow_tf32 = false\n[test]\npolicy = \"strict\"\n",
                )
                .unwrap();
            }
            "burn" | "standalone_precedence" | "set_precedence" => {
                std::fs::write(
                    dir.path().join("burn.toml"),
                    "[cubek.matmul]\nallow_tf32 = false\n",
                )
                .unwrap();
                if case == "standalone_precedence" {
                    std::fs::write(dir.path().join("cubek.toml"), "[matmul]\n").unwrap();
                }
            }
            _ => {}
        }
        // Also exercises walking up from a child directory.
        let cwd = dir.path().join("child");
        std::fs::create_dir(&cwd).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "configuration_child", "--nocapture"])
            .env("CUBEK_CONFIG_TEST_CASE", case)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{case}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn configuration_child() {
    let Ok(case) = std::env::var("CUBEK_CONFIG_TEST_CASE") else {
        return;
    };
    let expected = matches!(
        case.as_str(),
        "defaults" | "standalone_precedence" | "set_precedence"
    );
    if case == "set_precedence" {
        CubekConfig::set(CubekConfig::default());
    }
    if case == "concurrent_read" {
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| assert!(!MatmulConfig::get().allow_tf32));
            }
        });
    }
    assert_eq!(MatmulConfig::get().allow_tf32, expected);
    assert_eq!(CubekConfig::get().matmul.allow_tf32, expected);
    assert!(!CubekConfig::try_set(CubekConfig {
        matmul: MatmulConfig {
            allow_tf32: !expected,
        },
    }));
    assert_eq!(MatmulConfig::get().allow_tf32, expected);
}

#[test]
fn serialized_config_round_trips_and_rejects_invalid_precision() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "[matmul]\nallow_tf32 = false\n").unwrap();
    let config = CubekConfig::from_file_path(&path).unwrap();
    assert!(!config.matmul.allow_tf32);
    // save_default serializes the active config, despite its inherited name.
    CubekConfig::set(config);
    CubekConfig::save_default(&path).unwrap();
    assert!(
        !CubekConfig::from_file_path(&path)
            .unwrap()
            .matmul
            .allow_tf32
    );
    std::fs::write(&path, "[matmul]\nallow_tf32 = 'false'\n").unwrap();
    assert!(CubekConfig::from_file_path(&path).is_err());
}
