//! The shape of the crate's source, held to a ceiling: how much is public, how long functions
//! and files are, how many argument-count lints are silenced, and whether a retired word is back.
//!
//! Each ceiling is the number the tree had when the measure was added; a phase that lowers one
//! lowers the constant with it, and nothing raises one without saying why here.

use std::fs;
use std::path::{Path, PathBuf};

/// Public items declared anywhere in `src/`: types, traits, functions, constants, modules,
/// re-exports. `pub(crate)` and `pub(super)` are not public.
///
/// Counts what the crate compiles: a file no `mod` declares is not part of the surface, and one
/// left behind by a merge was inflating this by 33 until it was deleted. Phase 10 sets the
/// target the facade lands on.
///
/// `RowChunks`, how a tiled stage's block lays its rows down, raised it by one: a caller states it.
/// Its `CHUNK_BYTES`, what a padded row grows by, raised it by one more: a caller budgeting shared
/// memory counts it.
const PUB_ITEMS: usize = 763;
/// Functions whose body runs past this many lines.
const LONG_FN_LINES: usize = 60;
const LONG_FNS: usize = 36;
/// Files longer than this, tests included.
const LONG_FILE_LINES: usize = 500;
const LONG_FILES: usize = 11;
/// `#[allow(clippy::too_many_arguments)]` sites. The 27th came in with upstream's
/// shared-memory accumulator.
const TOO_MANY_ARGUMENTS: usize = 27;

/// Words the redesign retires, each checked as a whole identifier. A phase that deletes a concept
/// moves its word here, and the count must be zero from then on.
const RETIRED: &[&str] = &[
    "Fold",
    "FoldSeq",
    "Foldable",
    "ByAxis",
    "MAX_AXES",
    "const_coords",
    "last_cube_in",
    "unravel_const",
    "concat3",
    "within_2d",
    "AxisDistribution",
    "Distributed",
    "DistributedToUnits",
    "distribution",
    "distributes",
    "distributed",
];

#[test]
fn the_source_stays_under_its_ceilings() {
    let census = Census::new(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"));
    println!("{census}");
    assert!(
        census.pub_items <= PUB_ITEMS,
        "public items rose to {} (ceiling {PUB_ITEMS})",
        census.pub_items
    );
    assert!(
        census.long_fns.len() <= LONG_FNS,
        "functions over {LONG_FN_LINES} lines rose to {} (ceiling {LONG_FNS}): {:?}",
        census.long_fns.len(),
        census.long_fns
    );
    assert!(
        census.long_files.len() <= LONG_FILES,
        "files over {LONG_FILE_LINES} lines rose to {} (ceiling {LONG_FILES}): {:?}",
        census.long_files.len(),
        census.long_files
    );
    assert!(
        census.too_many_arguments <= TOO_MANY_ARGUMENTS,
        "too_many_arguments allowances rose to {} (ceiling {TOO_MANY_ARGUMENTS})",
        census.too_many_arguments
    );
    assert!(
        census.retired.is_empty(),
        "retired words are back: {:?}",
        census.retired
    );
}

/// What one pass over `src/` counted.
struct Census {
    pub_items: usize,
    long_fns: Vec<String>,
    long_files: Vec<String>,
    too_many_arguments: usize,
    retired: Vec<String>,
}

impl Census {
    fn new(src: &Path) -> Self {
        let mut census = Census {
            pub_items: 0,
            long_fns: Vec::new(),
            long_files: Vec::new(),
            too_many_arguments: 0,
            retired: Vec::new(),
        };
        let mut files = Vec::new();
        collect(src, &mut files);
        files.sort();
        for path in files {
            let text = fs::read_to_string(&path).expect("a source file reads");
            let name = path
                .strip_prefix(src)
                .unwrap_or(&path)
                .display()
                .to_string();
            census.file(&name, &text);
        }
        census
    }

    fn file(&mut self, name: &str, text: &str) {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() > LONG_FILE_LINES {
            self.long_files.push(format!("{name} ({})", lines.len()));
        }
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if is_pub_item(trimmed) {
                self.pub_items += 1;
            }
            if trimmed.contains("clippy::too_many_arguments") {
                self.too_many_arguments += 1;
            }
            for word in RETIRED {
                if has_identifier(strip_comment(line), word) {
                    self.retired.push(format!("{name}:{} {word}", i + 1));
                }
            }
            if let Some(fn_name) = fn_start(trimmed) {
                let body = body_lines(&lines, i);
                if body > LONG_FN_LINES {
                    self.long_fns
                        .push(format!("{name}:{} {fn_name} ({body})", i + 1));
                }
            }
        }
    }
}

impl std::fmt::Display for Census {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "public items: {}", self.pub_items)?;
        writeln!(
            f,
            "functions over {LONG_FN_LINES} lines: {}",
            self.long_fns.len()
        )?;
        for item in &self.long_fns {
            writeln!(f, "  {item}")?;
        }
        writeln!(
            f,
            "files over {LONG_FILE_LINES} lines: {}",
            self.long_files.len()
        )?;
        for item in &self.long_files {
            writeln!(f, "  {item}")?;
        }
        writeln!(
            f,
            "too_many_arguments allowances: {}",
            self.too_many_arguments
        )?;
        writeln!(f, "retired words present: {}", self.retired.len())
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("src is a directory") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// A `pub` declaration that is not restricted to the crate or a parent module.
fn is_pub_item(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("pub ") else {
        return false;
    };
    [
        "struct ",
        "enum ",
        "trait ",
        "fn ",
        "type ",
        "const ",
        "static ",
        "mod ",
        "use ",
        "unsafe fn ",
    ]
    .iter()
    .any(|kw| rest.starts_with(kw))
}

/// The name of a function whose declaration starts on this line, at any visibility.
fn fn_start(line: &str) -> Option<&str> {
    let mut rest = line;
    for prefix in [
        "pub(crate) ",
        "pub(super) ",
        "pub ",
        "unsafe ",
        "const ",
        "async ",
    ] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped;
        }
    }
    let rest = rest.strip_prefix("fn ")?;
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    Some(&rest[..end])
}

/// Lines from the declaration to the closing brace of its body, by brace depth. A declaration
/// with no body (a trait method) counts as one line.
fn body_lines(lines: &[&str], start: usize) -> usize {
    let mut depth = 0i32;
    let mut opened = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        let code = strip_comment(line);
        for c in code.chars() {
            match c {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if !opened && code.trim_end().ends_with(';') {
            return 1;
        }
        if opened && depth <= 0 {
            return offset + 1;
        }
    }
    lines.len() - start
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(at) => &line[..at],
        None => line,
    }
}

fn has_identifier(line: &str, word: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(word) {
        let start = from + at;
        let end = start + word.len();
        let before = start == 0 || !is_ident(bytes[start - 1]);
        let after = end == bytes.len() || !is_ident(bytes[end]);
        if before && after {
            return true;
        }
        from = end;
    }
    false
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
