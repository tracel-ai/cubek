//! `cubek.toml` loader.
//!
//! At process start we walk up from the current working directory looking
//! for a file named `cubek.toml`. The first match wins, the parsed result
//! is cached in a `OnceLock`, and every other module asks for the parts it
//! needs through this module.
//!
//! The parser is intentionally tiny — `cubek.toml` is a flat schema (one
//! `[section]` key per group, scalar values only). Anything more elaborate
//! belongs in a real TOML library; for now we accept exactly what the
//! example file documents and panic loudly on anything else.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::correctness::{TensorFilter, parse_tensor_filter};

/// Top-level configuration loaded from `cubek.toml`.
#[derive(Clone, Debug)]
pub struct CubekConfig {
    pub test: TestSection,
    pub print: PrintSection,
}

#[derive(Clone, Debug)]
pub struct TestSection {
    pub policy: TestPolicy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TestPolicy {
    Correct,
    Strict,
    FailIfRun,
}

#[derive(Clone, Debug)]
pub struct PrintSection {
    pub enabled: bool,
    pub view: PrintView,
    pub force_fail: bool,
    pub fail_only: bool,
    pub show_expected: bool,
    pub filter: TensorFilter,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrintView {
    Table,
    Lines,
}

impl Default for CubekConfig {
    fn default() -> Self {
        Self {
            test: TestSection {
                policy: TestPolicy::Correct,
            },
            print: PrintSection {
                enabled: false,
                view: PrintView::Table,
                force_fail: true,
                fail_only: false,
                show_expected: false,
                filter: Vec::new(),
            },
        }
    }
}

/// Returns the active config. Cached on first call.
pub fn config() -> &'static CubekConfig {
    static CACHE: OnceLock<CubekConfig> = OnceLock::new();
    CACHE.get_or_init(load_config)
}

fn load_config() -> CubekConfig {
    let Some(path) = find_cubek_toml() else {
        return CubekConfig::default();
    };
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    parse_cubek_toml(&text)
        .unwrap_or_else(|e| panic!("invalid cubek.toml ({}): {e}", path.display()))
}

fn find_cubek_toml() -> Option<PathBuf> {
    let mut cur = std::env::current_dir().ok()?;
    loop {
        let candidate = cur.join("cubek.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !cur.pop() {
            return None;
        }
    }
}

// ---------- minimal flat-section TOML parser ----------

type Sections = HashMap<String, HashMap<String, String>>;

fn parse_cubek_toml(text: &str) -> Result<CubekConfig, String> {
    let sections = parse_sections(text)?;

    let mut cfg = CubekConfig::default();

    if let Some(map) = sections.get("test") {
        cfg.test.policy = match get_string(map, "policy")?.as_deref() {
            None | Some("correct") => TestPolicy::Correct,
            Some("strict") => TestPolicy::Strict,
            Some("fail-if-run") => TestPolicy::FailIfRun,
            Some(other) => {
                return Err(format!(
                    "[test] policy='{}': expected one of \"correct\", \"strict\", \"fail-if-run\"",
                    other
                ));
            }
        };
        reject_unknown_keys("test", map, &["policy"])?;
    }

    if let Some(map) = sections.get("print") {
        let enabled = get_bool(map, "enabled")?.unwrap_or(false);
        let view = match get_string(map, "view")?.as_deref() {
            None | Some("table") => PrintView::Table,
            Some("lines") => PrintView::Lines,
            Some(other) => {
                return Err(format!(
                    "[print] view='{}': expected \"table\" or \"lines\"",
                    other
                ));
            }
        };
        let force_fail = get_bool(map, "force-fail")?.unwrap_or(true);
        let fail_only = get_bool(map, "fail-only")?.unwrap_or(false);
        let show_expected = get_bool(map, "show-expected")?.unwrap_or(false);
        let filter_str = get_string(map, "filter")?.unwrap_or_default();
        let filter = if filter_str.is_empty() {
            Vec::new()
        } else {
            parse_tensor_filter(&filter_str)
                .map_err(|e| format!("[print] filter='{}': {}", filter_str, e))?
        };

        cfg.print = PrintSection {
            enabled,
            view,
            force_fail,
            fail_only,
            show_expected,
            filter,
        };

        reject_unknown_keys(
            "print",
            map,
            &[
                "enabled",
                "view",
                "force-fail",
                "fail-only",
                "show-expected",
                "filter",
            ],
        )?;
    }

    for sec in sections.keys() {
        if sec != "test" && sec != "print" {
            return Err(format!("unknown section [{}]", sec));
        }
    }

    Ok(cfg)
}

fn parse_sections(text: &str) -> Result<Sections, String> {
    let mut sections: Sections = HashMap::new();
    let mut current: Option<String> = None;

    for (line_no, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[')
            && let Some(name) = rest.strip_suffix(']')
        {
            let name = name.trim();
            if name.is_empty() || name.contains('.') {
                return Err(format!(
                    "line {}: section name '[{}]' must be a single identifier",
                    line_no + 1,
                    name
                ));
            }
            sections.entry(name.to_string()).or_default();
            current = Some(name.to_string());
            continue;
        }

        let Some(section) = current.as_ref() else {
            return Err(format!(
                "line {}: key '{}' before any [section]",
                line_no + 1,
                line
            ));
        };

        let Some((k, v)) = line.split_once('=') else {
            return Err(format!(
                "line {}: expected `key = value`, got '{}'",
                line_no + 1,
                line
            ));
        };
        let key = k.trim().to_string();
        let val = v.trim().to_string();
        sections.get_mut(section).unwrap().insert(key, val);
    }

    Ok(sections)
}

fn strip_comment(line: &str) -> &str {
    // TOML allows '#' anywhere outside a string. We only support unquoted
    // scalars and strings without '#' inside; that's enough for cubek.toml.
    let mut in_string = false;
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_string = !in_string,
            b'#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

fn get_string(map: &HashMap<String, String>, key: &str) -> Result<Option<String>, String> {
    let Some(raw) = map.get(key) else {
        return Ok(None);
    };
    let s = unquote(raw)
        .ok_or_else(|| format!("key '{}' must be a quoted string, got `{}`", key, raw))?;
    Ok(Some(s))
}

fn get_bool(map: &HashMap<String, String>, key: &str) -> Result<Option<bool>, String> {
    let Some(raw) = map.get(key) else {
        return Ok(None);
    };
    match raw.as_str() {
        "true" => Ok(Some(true)),
        "false" => Ok(Some(false)),
        other => Err(format!(
            "key '{}' must be `true` or `false`, got `{}`",
            key, other
        )),
    }
}

fn unquote(s: &str) -> Option<String> {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

fn reject_unknown_keys(
    section: &str,
    map: &HashMap<String, String>,
    known: &[&str],
) -> Result<(), String> {
    for k in map.keys() {
        if !known.contains(&k.as_str()) {
            return Err(format!(
                "[{}] unknown key '{}'. Known: {}",
                section,
                k,
                known.join(", ")
            ));
        }
    }
    Ok(())
}

// ---------- unit tests for the parser ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_example() {
        let text = r#"
[test]
policy = "strict"

[print]
enabled = true
view = "lines"
force-fail = false
fail-only = true
show-expected = true
filter = "0,1-2"
"#;
        let cfg = parse_cubek_toml(text).unwrap();
        assert_eq!(cfg.test.policy, TestPolicy::Strict);
        assert!(cfg.print.enabled);
        assert_eq!(cfg.print.view, PrintView::Lines);
        assert!(!cfg.print.force_fail);
        assert!(cfg.print.fail_only);
        assert!(cfg.print.show_expected);
        assert_eq!(cfg.print.filter.len(), 2);
    }

    #[test]
    fn empty_file_gives_defaults() {
        let cfg = parse_cubek_toml("").unwrap();
        assert_eq!(cfg.test.policy, TestPolicy::Correct);
        assert!(!cfg.print.enabled);
        assert_eq!(cfg.print.view, PrintView::Table);
    }

    #[test]
    fn rejects_unknown_section() {
        let err = parse_cubek_toml("[bogus]\nx=1\n").unwrap_err();
        assert!(err.contains("unknown section"), "{}", err);
    }

    #[test]
    fn rejects_unknown_key() {
        let err = parse_cubek_toml("[print]\nbogus = true\n").unwrap_err();
        assert!(err.contains("unknown key"), "{}", err);
    }

    #[test]
    fn rejects_bad_policy() {
        let err = parse_cubek_toml("[test]\npolicy = \"loose\"\n").unwrap_err();
        assert!(err.contains("policy"), "{}", err);
    }

    #[test]
    fn rejects_unquoted_string() {
        let err = parse_cubek_toml("[print]\nview = table\n").unwrap_err();
        assert!(err.contains("quoted string"), "{}", err);
    }

    #[test]
    fn rejects_show_delta_key() {
        // We removed `show-delta`/`show-epsilon` — they should now be
        // unknown keys, since lines view always shows them and table view
        // never does.
        let err = parse_cubek_toml("[print]\nshow-delta = true\n").unwrap_err();
        assert!(err.contains("unknown key"), "{}", err);
    }
}
