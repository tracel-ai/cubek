mod commands;

#[macro_use]
extern crate log;

use tracel_xtask::prelude::*;

#[macros::base_commands(
    Bump,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Build cubecl in different modes.
    Check(CheckCmdArgs),
    /// Test cubecl.
    Test(commands::test::CubeKTestCmdArgs),
    /// Profile kernels.
    Profile(commands::profile::ProfileArgs),
}

fn main() -> anyhow::Result<()> {
    let (args, env) = init_xtask::<Command>(parse_args::<Command>()?)?;
    match args.command {
        Command::Test(cmd_args) => commands::test::handle_command(cmd_args, env, args.context),
        Command::Profile(cmd_args) => cmd_args.run(),
        Command::Check(cmd_args) => {
            base_commands::check::handle_command(cmd_args, env, args.context)
        }
        _ => dispatch_base_commands(args, env),
    }?;
    Ok(())
}
