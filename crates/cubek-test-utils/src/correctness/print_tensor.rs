use crate::HostData;
use crate::config::config;
use crate::correctness::render::print_tensors as render;

/// Pretty-print one or two tensors through the unified renderer.
///
/// Two tensors of the same shape are rendered as a colored diff (cells red
/// when `|a - b| > epsilon`, green otherwise). One tensor renders without
/// color. Tensors with different shapes (or rank) are silently skipped.
///
/// No-op when `[print] enabled = false` in `cubek.toml`.
///
/// # Examples
///
/// ```ignore
/// use cubek_test_utils::print_tensors;
///
/// // Single tensor — just pretty-print, no color.
/// print_tensors("input", &[&host], None);
///
/// // Two tensors of the same shape — colored diff. Same path is used by
/// // `assert_equals_approx` for actual-vs-expected.
/// print_tensors("a vs b", &[&a, &b], Some(1e-3));
/// ```
pub fn print_tensors(label: &str, tensors: &[&HostData], epsilon: Option<f32>) {
    render(&config().print, label, tensors, epsilon);
}

/// Backwards-compatible single-tensor helper. Delegates to [`print_tensors`].
pub fn print_tensor(label: &str, host: &HostData) {
    print_tensors(label, &[host], None);
}
