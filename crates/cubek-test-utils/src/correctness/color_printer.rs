use crate::correctness::{CompareVisitor, ElemStatus, WrongStatus};

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const RESET: &str = "\x1b[0m";

pub(crate) struct ColorPrinter {
    pub filter: TensorFilter,
    indent: usize,
}

impl ColorPrinter {
    pub fn new(filter: TensorFilter) -> Self {
        Self { filter, indent: 0 }
    }

    fn should_print(&self, index: &[usize]) -> bool {
        index_matches_filter(index, &self.filter)
    }
}

impl CompareVisitor for ColorPrinter {
    fn visit(&mut self, index: &[usize], status: ElemStatus) {
        if !self.should_print(index) {
            return;
        }

        let idx = format!(
            "({})",
            index
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        match status {
            ElemStatus::Correct { got } => {
                println!(
                    "{}{}: {}{}{}",
                    " ".repeat(self.indent),
                    idx,
                    GREEN,
                    got,
                    RESET
                );
            }
            ElemStatus::Wrong(wrong) => match wrong {
                WrongStatus::GotWrongValue {
                    got,
                    expected,
                    diff,
                    epsilon,
                } => {
                    println!(
                        "{}{}: {}Got {}, expected {}, diff={}>{}{}",
                        " ".repeat(self.indent),
                        idx,
                        RED,
                        got,
                        expected,
                        diff,
                        epsilon,
                        RESET
                    );
                }
                WrongStatus::ExpectedNan { got } => {
                    println!(
                        "{}{}: {}Got {}, expected NaN{}",
                        " ".repeat(self.indent),
                        idx,
                        RED,
                        got,
                        RESET
                    );
                }
                WrongStatus::GotNan { expected } => {
                    println!(
                        "{}{}: {}Got NaN, expected {}{}",
                        " ".repeat(self.indent),
                        idx,
                        RED,
                        expected,
                        RESET
                    );
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum DimFilter {
    Any,
    Exact(usize),
    Range { start: usize, end: usize },
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
