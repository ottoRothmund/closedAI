// SPDX-License-Identifier: AGPL-3.0-or-later

mod run;

use clap::{Parser, Subcommand};

/// closedAI — collectively run one open model.
#[derive(Parser)]
#[command(name = "closedai", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load a model and generate text on this machine.
    Run(run::RunArgs),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => run::run(args),
    }
}
