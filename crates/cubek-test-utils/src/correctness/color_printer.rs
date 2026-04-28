//! Filter types shared by `CUBE_TEST_MODE`, `assert_equals_approx_in_slice`,
//! `print_tensor`, and `pretty_print_slice`. Per-element rendering moved to
//! `correctness::render`; this module is now just the filter language.

#[derive(Debug, Clone)]
pub enum DimFilter {
    /// Wildcard: match any index along this dimension.
    Any,
    /// Match a single exact index.
    Exact(usize),
    /// Inclusive range: matches `start..=end`.
    /// (`CUBE_TEST_MODE`'s `M-K` filter and the parser produce this variant.)
    Range { start: usize, end: usize },
}

impl From<std::ops::Range<usize>> for DimFilter {
    /// Convert a half-open `Range<usize>` (`start..end`) into an inclusive
    /// `DimFilter::Range`. An empty range is converted into a filter that
    /// matches nothing.
    fn from(r: std::ops::Range<usize>) -> Self {
        if r.start >= r.end {
            // Empty range — produce a filter that excludes every index.
            DimFilter::Exact(usize::MAX)
        } else {
            DimFilter::Range {
                start: r.start,
                end: r.end - 1,
            }
        }
    }
}

pub type TensorFilter = Vec<DimFilter>;

pub fn parse_tensor_filter(s: &str) -> Result<TensorFilter, String> {
    if s.is_empty() {
        return Ok(vec![]);
    }

    let mut filters = Vec::new();

    for part in s.split(',') {
        let f = if part == "." {
            DimFilter::Any
        } else if let Some((a, b)) = part.split_once('-') {
            DimFilter::Range {
                start: a.parse().map_err(|_| format!("Invalid number: {}", a))?,
                end: b.parse().map_err(|_| format!("Invalid number: {}", b))?,
            }
        } else {
            DimFilter::Exact(
                part.parse()
                    .map_err(|_| format!("Invalid filter token: {}", part))?,
            )
        };

        filters.push(f);
    }

    Ok(filters)
}

pub(crate) fn index_matches_filter(index: &[usize], filter: &TensorFilter) -> bool {
    for (dim, idx) in index.iter().copied().enumerate() {
        let f = filter.get(dim).unwrap_or(&DimFilter::Any);

        match f {
            DimFilter::Any => {}
            DimFilter::Exact(v) => {
                if idx != *v {
                    return false;
                }
            }
            DimFilter::Range { start, end } => {
                if idx < *start || idx > *end {
                    return false;
                }
            }
        }
    }
    true
}
