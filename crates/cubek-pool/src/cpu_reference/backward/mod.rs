mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool_backward;
pub use avg_pool::run_avg_pool_backward;
pub use max_pool::run_max_pool_backward;
