//! The crate's error type. Every failure names what it was reading, so a bad
//! checkpoint or config fails loud with a pointer to the problem.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// A file could not be read.
    Io {
        path: String,
        source: std::io::Error,
    },
    /// Malformed JSON, or JSON of the wrong shape (missing key, wrong type).
    Json(String),
    /// `config.json` parsed but breaks the invariants the engine relies on.
    /// Every violated invariant is listed.
    Invariant(Vec<String>),
    /// A malformed safetensors file or header.
    Safetensors(String),
    /// A checkpoint whose tensors do not match the catalog for its format.
    Catalog(String),
    /// A tensor-parallel split that cannot be made exactly.
    Slicing(String),
    /// A memory plan that cannot be satisfied.
    Plan(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn io_err(path: &std::path::Path, source: std::io::Error) -> Error {
    Error::Io {
        path: path.display().to_string(),
        source,
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{path}: {source}"),
            Error::Json(m) => write!(f, "json: {m}"),
            Error::Invariant(list) => {
                write!(f, "config breaks {} invariant(s):", list.len())?;
                for m in list {
                    write!(f, "\n  - {m}")?;
                }
                Ok(())
            }
            Error::Safetensors(m) => write!(f, "safetensors: {m}"),
            Error::Catalog(m) => write!(f, "catalog: {m}"),
            Error::Slicing(m) => write!(f, "slicing: {m}"),
            Error::Plan(m) => write!(f, "plan: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
