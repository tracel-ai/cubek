//! Startup cases run in separate processes because CubekConfig is immutable.

#![cfg(all(
    feature = "std",
    any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos",
        target_os = "android"
    )
))]

use cubek_std::config::{CubekConfig, NanPolicy, RuntimeConfig, nan_policy};

#[test]
fn startup_cases() {
    for case in [
        "default",
        "propagate",
        "file",
        "burn_file",
        "standalone_over_burn",
        "set_over_file",
    ] {
        let dir = tempfile::tempdir().unwrap();
        // Prevent finding a parent application's configuration.
        std::fs::write(dir.path().join("burn.toml"), "[cubek]\n").unwrap();
        match case {
            "file" | "set_over_file" => {
                std::fs::write(
                    dir.path().join("cubek.toml"),
                    "[numerics]\nnan_policy = \"propagate\"\n",
                )
                .unwrap();
            }
            "burn_file" | "standalone_over_burn" => {
                std::fs::write(
                    dir.path().join("burn.toml"),
                    "[fusion]\n[cubek.numerics]\nnan_policy = \"propagate\"\n",
                )
                .unwrap();
                if case == "standalone_over_burn" {
                    std::fs::write(
                        dir.path().join("cubek.toml"),
                        "[numerics]\nnan_policy = \"native\"\n",
                    )
                    .unwrap();
                }
            }
            _ => {}
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "startup_child", "--ignored", "--nocapture"])
            .env("CUBEK_NUMERICS_TEST_CASE", case)
            .current_dir(dir.path())
            .status()
            .unwrap();
        assert!(status.success(), "startup case {case}");
    }
}

#[test]
#[ignore = "invoked in isolated processes by startup_cases"]
fn startup_child() {
    let case = std::env::var("CUBEK_NUMERICS_TEST_CASE").unwrap();
    let expected = match case.as_str() {
        "default" | "standalone_over_burn" => NanPolicy::Native,
        "set_over_file" => {
            CubekConfig::set(CubekConfig::default().with_nan_policy(NanPolicy::Native));
            NanPolicy::Native
        }
        "propagate" => {
            CubekConfig::set(CubekConfig::default().with_nan_policy(NanPolicy::Propagate));
            NanPolicy::Propagate
        }
        "file" | "burn_file" => NanPolicy::Propagate,
        _ => panic!("unknown startup case"),
    };
    assert_eq!(nan_policy(), expected);
    assert_eq!(CubekConfig::get().numerics.nan_policy, expected);
    assert_eq!(nan_policy(), expected);
}

#[test]
fn parses_policy_and_rejects_unknown_values() {
    for (value, expected) in [
        ("native", NanPolicy::Native),
        ("propagate", NanPolicy::Propagate),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            format!("[numerics]\nnan_policy = \"{value}\"\n"),
        )
        .unwrap();
        let config = CubekConfig::from_file_path(file.path()).unwrap();
        assert_eq!(config.numerics.nan_policy, expected);
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), "[numerics]\nnan_policy = \"ignore\"\n").unwrap();
    assert!(CubekConfig::from_file_path(file.path()).is_err());
}
