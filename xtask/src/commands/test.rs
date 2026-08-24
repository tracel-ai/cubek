use tracel_xtask::prelude::*;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct CubeKTestCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    args: CubeKTestCmdArgs,
    _env: Environment,
    _context: Context,
) -> anyhow::Result<()> {
    let backends: &[&str] = if args.ci {
        &["cubecl/cpu"]
    } else {
        &["cubecl/wgpu", "cubecl/cpu"]
    };
    for backend in backends {
        build_helpers::custom_crates_tests(
            vec![
                "cubek-attention",
                "cubek-convolution",
                "cubek-fft",
                "cubek-interpolate",
                "cubek-matmul",
                "cubek-pool",
                "cubek-quant",
                "cubek-random",
                "cubek-reduce",
                "cubek-resample",
                "cubek-std",
                "cubek-test-utils",
                "cubek-tile",
            ],
            vec!["--features", backend],
            None,
            None,
            &format!("Test on backend {backend:?}"),
        )?;

        // The seam: cubek-matmul must pass with either architecture alone, or the cfg
        // attributes rot. The tiled-only pass is the one that proves the goal; the
        // multi-level-only pass proves multi-level never grew a tile-DSL dependency.
        for architecture in ["tiled", "multi-level"] {
            let features = format!("std,{architecture},{backend}");
            build_helpers::custom_crates_tests(
                vec!["cubek-matmul"],
                vec!["--no-default-features", "--features", &features],
                None,
                None,
                &format!("Test cubek-matmul on {architecture:?} alone, backend {backend:?}"),
            )?;
        }
    }
    Ok(())
}
