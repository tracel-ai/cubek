//! Unified tensor renderer.
//!
//! One entry point: [`print_tensors`]. It takes a slice of one or two
//! `HostData` references and renders them according to the active
//! [`PrintSection`]. There is no separate "diff" path — comparing actual to
//! expected and pretty-printing two same-shape tensors are the same call.
//!
//! Rules:
//! - If `cfg.enabled` is `false`, this is a no-op.
//! - If two tensors are passed and their ranks or shapes don't match, this
//!   is a no-op too (we don't know how to align them).
//! - With one tensor, color is off and Δ/ε are meaningless. With two, each
//!   cell is colored by the per-element comparison against `epsilon`
//!   (green = within ε, red = beyond).

use crate::config::{PrintSection, PrintView};
use crate::{DimFilter, HostData, TensorFilter};

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

/// Render one or two tensors. Caller passes `tensors.len() ∈ {1, 2}`. Two
/// tensors implies a diff: cells colored by `|a - b| > epsilon`.
///
/// Silently skips if:
/// - `cfg.enabled` is false,
/// - `tensors.len()` is 0 or > 2,
/// - two tensors have different ranks or shapes,
/// - the configured filter's rank doesn't match the tensor rank.
pub fn print_tensors(cfg: &PrintSection, label: &str, tensors: &[&HostData], epsilon: Option<f32>) {
    if !cfg.enabled || tensors.is_empty() || tensors.len() > 2 {
        return;
    }
    let primary = tensors[0];
    let other = tensors.get(1).copied();

    if let Some(b) = other
        && (b.shape.rank() != primary.shape.rank()
            || b.shape.as_slice() != primary.shape.as_slice())
    {
        return;
    }

    if !cfg.filter.is_empty() && cfg.filter.len() != primary.shape.rank() {
        return;
    }

    let filter = if cfg.filter.is_empty() {
        None
    } else {
        Some(cfg.filter.clone())
    };
    // ε isn't in the header — it's per-cell (each cell uses
    // `max(epsilon, epsilon * |expected|)`), so a single number would lie.
    println!("=== {label}  shape={:?} ===", primary.shape);

    let eps = epsilon.unwrap_or(0.0);
    match cfg.view {
        PrintView::Table => render_table(cfg, primary, other, eps, filter.as_ref()),
        PrintView::Lines => render_lines(cfg, primary, other, eps, filter.as_ref()),
    }
}

// ---------- table view ----------

fn render_table(
    cfg: &PrintSection,
    a: &HostData,
    b: Option<&HostData>,
    epsilon: f32,
    filter: Option<&TensorFilter>,
) {
    let rank = a.shape.rank();
    if rank == 0 {
        return;
    }

    let cell = |full: &[usize]| -> String {
        let av = a.get_f32(full);
        match b {
            None => format_value(av),
            Some(rhs) => {
                let bv = rhs.get_f32(full);
                let cell_eps = (epsilon * bv).abs().max(epsilon);
                let is_wrong = compare_pair(av, bv, cell_eps);
                if cfg.fail_only && !is_wrong {
                    // Blank out matching cells — the table layout still
                    // pads them, so the surviving red cells stay aligned.
                    return String::new();
                }
                let color = if is_wrong { RED } else { GREEN };
                let text = if cfg.show_expected {
                    format!("{}/{}", format_value(av), format_value(bv))
                } else {
                    format_value(av)
                };
                format!("{color}{text}{RESET}")
            }
        }
    };

    if rank == 1 {
        let col_filter = filter.and_then(|f| f.first());
        let cols = axis_indices(col_filter, a.shape[0]);
        let rows = vec![0usize];
        print_table(&rows, &cols, |_, c| cell(&[c]));
    } else if rank == 2 {
        let row_filter = filter.and_then(|f| f.first());
        let col_filter = filter.and_then(|f| f.get(1));
        let rows = axis_indices(row_filter, a.shape[0]);
        let cols = axis_indices(col_filter, a.shape[1]);
        print_table(&rows, &cols, |r, c| cell(&[r, c]));
    } else {
        let leading_dims = rank - 2;
        let row_dim = a.shape[rank - 2];
        let col_dim = a.shape[rank - 1];
        let row_indices = axis_indices(filter.and_then(|f| f.get(rank - 2)), row_dim);
        let col_indices = axis_indices(filter.and_then(|f| f.get(rank - 1)), col_dim);

        let mut leading = vec![0usize; leading_dims];
        let mut first = true;
        loop {
            let print_this = filter
                .map(|f| leading_indices_match(&leading, f))
                .unwrap_or(true);
            if print_this {
                if !first {
                    println!();
                }
                println!("{}:", format_leading_label(&leading, rank));
                print_table(&row_indices, &col_indices, |r, c| {
                    let mut full = leading.clone();
                    full.push(r);
                    full.push(c);
                    cell(&full)
                });
                first = false;
            }
            if !increment_lex(&mut leading, &a.shape.as_slice()[..leading_dims]) {
                break;
            }
        }
    }
}

// ---------- lines view ----------

struct LineRow {
    idx: String,
    primary: String,
    other: Option<String>,
    /// Δ and ε strings, present iff this is a 2-tensor (diff) row.
    delta: Option<String>,
    epsilon: Option<String>,
    is_wrong: bool,
}

fn render_lines(
    cfg: &PrintSection,
    a: &HostData,
    b: Option<&HostData>,
    epsilon: f32,
    filter: Option<&TensorFilter>,
) {
    let mut rows: Vec<LineRow> = Vec::new();
    for idx in a.iter_indices() {
        if let Some(f) = filter
            && !index_matches(&idx, f)
        {
            continue;
        }
        let av = a.get_f32(&idx);
        match b {
            None => {
                rows.push(LineRow {
                    idx: format_index(&idx),
                    primary: format_value(av),
                    other: None,
                    delta: None,
                    epsilon: None,
                    is_wrong: false,
                });
            }
            Some(rhs) => {
                let bv = rhs.get_f32(&idx);
                let cell_eps = (epsilon * bv).abs().max(epsilon);
                let delta = (av - bv).abs();
                let is_wrong = compare_pair(av, bv, cell_eps);
                if cfg.fail_only && !is_wrong {
                    continue;
                }
                rows.push(LineRow {
                    idx: format_index(&idx),
                    primary: format_value(av),
                    other: Some(format_value(bv)),
                    // Δ and ε are always shown in lines view. Suppressing
                    // them is what the `table` view is for.
                    delta: Some(format_value(delta)),
                    epsilon: Some(format_value(cell_eps)),
                    is_wrong,
                });
            }
        }
    }

    if rows.is_empty() {
        return;
    }

    // Column widths.
    let idx_w = rows.iter().map(|r| r.idx.len()).max().unwrap_or(0);
    let pri_w = rows.iter().map(|r| r.primary.len()).max().unwrap_or(0);
    let oth_w = rows
        .iter()
        .filter_map(|r| r.other.as_ref().map(|s| s.len()))
        .max()
        .unwrap_or(0);
    let dlt_w = rows
        .iter()
        .filter_map(|r| r.delta.as_ref().map(|s| s.len()))
        .max()
        .unwrap_or(0);
    let eps_w = rows
        .iter()
        .filter_map(|r| r.epsilon.as_ref().map(|s| s.len()))
        .max()
        .unwrap_or(0);

    let two = b.is_some();
    let primary_label = if two { "got" } else { "value" };

    // Header — Δ / ε / status are always present in 2-tensor lines view.
    let mut header = format!("{:>idx_w$} | {:>pri_w$}", "index", primary_label);
    if two {
        header.push_str(&format!(
            " | {:>oth_w$} | {:>dlt_w$} | {:>eps_w$} | status",
            "expected", "Δ", "ε",
        ));
    }
    println!("{header}");

    let mut total = idx_w + pri_w + 3;
    if two {
        total += oth_w + 3 + dlt_w + 3 + eps_w + 3 + " | status".len();
    }
    println!("{}", "-".repeat(total));

    for r in rows {
        let color = if two {
            if r.is_wrong { RED } else { GREEN }
        } else {
            ""
        };
        let reset = if two { RESET } else { "" };
        let mut line = format!("{}{:>idx_w$} | {:>pri_w$}", color, r.idx, r.primary,);
        if let Some(o) = r.other.as_ref() {
            line.push_str(&format!(" | {:>oth_w$}", o));
        }
        if let Some(d) = r.delta.as_ref() {
            line.push_str(&format!(" | {:>dlt_w$}", d));
        }
        if let Some(e) = r.epsilon.as_ref() {
            line.push_str(&format!(" | {:>eps_w$}", e));
        }
        if two {
            let status = if r.is_wrong { "FAIL" } else { "ok" };
            line.push_str(&format!(" | {status}"));
        }
        line.push_str(reset);
        println!("{line}");
    }
}

// ---------- shared helpers ----------

fn format_value(v: f32) -> String {
    format!("{:.6}", v)
}

fn format_index(idx: &[usize]) -> String {
    format!(
        "[{}]",
        idx.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Returns `true` if `(a, b)` is a mismatch under `epsilon`. Mirrors
/// `assert_equals_approx`'s NaN/Inf semantics.
fn compare_pair(a: f32, b: f32, epsilon: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return false;
    }
    if a.is_nan() || b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() != b.signum();
    }
    (a - b).abs() > epsilon
}

fn axis_indices(f: Option<&DimFilter>, dim_size: usize) -> Vec<usize> {
    match f {
        None | Some(DimFilter::Any) => (0..dim_size).collect(),
        Some(DimFilter::Exact(v)) => {
            if *v < dim_size {
                vec![*v]
            } else {
                Vec::new()
            }
        }
        Some(DimFilter::Range { start, end }) => {
            if *start >= dim_size {
                Vec::new()
            } else {
                (*start..=(*end).min(dim_size.saturating_sub(1))).collect()
            }
        }
    }
}

fn index_matches(index: &[usize], filter: &TensorFilter) -> bool {
    for (dim, idx) in index.iter().enumerate() {
        let f = filter.get(dim).unwrap_or(&DimFilter::Any);
        match f {
            DimFilter::Any => {}
            DimFilter::Exact(v) => {
                if idx != v {
                    return false;
                }
            }
            DimFilter::Range { start, end } => {
                if idx < start || idx > end {
                    return false;
                }
            }
        }
    }
    true
}

fn leading_indices_match(leading: &[usize], filter: &TensorFilter) -> bool {
    index_matches(leading, filter)
}

fn increment_lex(idx: &mut [usize], bounds: &[usize]) -> bool {
    if idx.is_empty() {
        return false;
    }
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] < bounds[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

fn format_leading_label(leading: &[usize], rank: usize) -> String {
    let mut parts: Vec<String> = leading.iter().map(|i| i.to_string()).collect();
    for _ in 0..(rank - leading.len()) {
        parts.push("*".to_string());
    }
    format!("[{}]", parts.join(", "))
}

fn print_table<F>(rows: &[usize], cols: &[usize], mut cell: F)
where
    F: FnMut(usize, usize) -> String,
{
    if rows.is_empty() || cols.is_empty() {
        return;
    }
    let mut max_width = 0;
    for &r in rows {
        for &c in cols {
            max_width = max_width.max(visible_width(&cell(r, c)));
        }
    }
    let label_width = cols.iter().map(|c| c.to_string().len()).max().unwrap_or(0);
    max_width = max_width.max(label_width).max(2);

    let row_label_width = rows
        .iter()
        .map(|r| r.to_string().len())
        .max()
        .unwrap_or(0)
        .max(3);

    let mut s = String::new();
    s.push_str(&format!("{:>width$} |", "", width = row_label_width));
    for &c in cols {
        s.push_str(&format!(" {:>width$}", c, width = max_width));
    }
    s.push('\n');
    s.push_str(&"-".repeat(row_label_width + 1));
    s.push('+');
    for _ in cols {
        s.push_str(&"-".repeat(max_width + 1));
    }
    s.push('\n');

    for &r in rows {
        s.push_str(&format!("{:>width$} |", r, width = row_label_width));
        for &c in cols {
            let raw = cell(r, c);
            let pad = max_width.saturating_sub(visible_width(&raw));
            s.push(' ');
            for _ in 0..pad {
                s.push(' ');
            }
            s.push_str(&raw);
        }
        s.push('\n');
    }
    print!("{s}");
}

/// `len()` ignoring ANSI escapes, so colored cells still align with plain ones.
fn visible_width(s: &str) -> usize {
    let mut n = 0;
    let mut iter = s.chars();
    while let Some(ch) = iter.next() {
        if ch == '\x1b' {
            for c in iter.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            n += 1;
        }
    }
    n
}
