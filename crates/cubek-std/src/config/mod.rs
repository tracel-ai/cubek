//! Runtime defaults for CubeK kernels and applications using them.
//!
//! Read `[numerics]` from `cubek.toml`, or `[cubek.numerics]` from `burn.toml`.
//! Configuration is immutable after its first read; explicit operation configurations
//! continue to determine the behavior of individual kernel launches.

use cubecl_environment::sync::{Arc, Mutex};

/// Numerical contracts for selected floating-point operations.
pub mod numerics;

pub use cubecl_environment::config::RuntimeConfig;
pub use numerics::{NanPolicy, NumericsConfig, nan_policy};

static CUBEK_GLOBAL_CONFIG: Mutex<Option<Arc<CubekConfig>>> = Mutex::new(None);

/// Runtime defaults shared by CubeK and its consumers.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CubekConfig {
    /// Configuration for numerical contracts.
    pub numerics: NumericsConfig,
}

impl CubekConfig {
    /// Selects the default NaN policy for covered operations.
    pub fn with_nan_policy(mut self, policy: NanPolicy) -> Self {
        self.numerics.nan_policy = policy;
        self
    }
}

impl RuntimeConfig for CubekConfig {
    fn on_loaded(&self) {
        numerics::publish_nan_policy(self.numerics.nan_policy);
    }

    fn storage() -> &'static Mutex<Option<Arc<Self>>> {
        &CUBEK_GLOBAL_CONFIG
    }

    fn file_names() -> &'static [&'static str] {
        &["cubek.toml", "CubeK.toml"]
    }

    fn section_file_names() -> &'static [(&'static str, &'static str)] {
        &[("burn.toml", "cubek"), ("Burn.toml", "cubek")]
    }
}
