//! Process-wide defaults for CubeK kernels.
//!
//! Configuration is loaded lazily from `cubek.toml` (or `CubeK.toml`), or the
//! `[cubek]` section of `burn.toml` (or `Burn.toml`). The search walks up from the
//! working directory, preferring standalone files at each directory level.
//! Missing fields use their defaults.
//!
//! To configure programmatically, import [`RuntimeConfig`] and call
//! [`CubekConfig::set`] before any kernel selection or configuration read.
//! Once initialized, the configuration cannot change. Remote execution uses the
//! configuration of the process executing the kernels.

use cubecl_environment::sync::{Arc, LazyLock, Mutex};

pub use cubecl_environment::config::RuntimeConfig;

static CUBEK_GLOBAL_CONFIG: Mutex<Option<Arc<CubekConfig>>> = Mutex::new(None);

/// Runtime defaults shared by CubeK and its consumers.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CubekConfig {
    /// Defaults for matrix multiplication.
    pub matmul: MatmulConfig,
}

/// Precision choices made automatically by matrix multiplication kernels.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MatmulConfig {
    /// Allow F32 inputs to be rounded to TF32 for accelerated multiplication.
    ///
    /// Defaults to `true`, preserving the existing selection behavior. Setting
    /// this to `false` keeps automatic F32 matmul selection in F32, potentially
    /// falling back to slower kernels. A forced accelerator strategy may then
    /// be unavailable.
    ///
    /// This does not change explicitly requested TF32 or `flex32` inputs,
    /// convolution's own precision choices, compiler fast-math flags, reduction
    /// order, or accumulation precision. It does not promise bitwise equality
    /// with other frameworks.
    pub allow_tf32: bool,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self { allow_tf32: true }
    }
}

impl MatmulConfig {
    /// Returns the process's matmul defaults, initializing [`CubekConfig`] if
    /// needed. Subsequent reads avoid the global configuration mutex.
    pub fn get() -> &'static Self {
        static CONFIG: LazyLock<MatmulConfig> = LazyLock::new(|| CubekConfig::get().matmul);
        &CONFIG
    }
}

impl RuntimeConfig for CubekConfig {
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
