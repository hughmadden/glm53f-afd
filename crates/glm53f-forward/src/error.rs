//! The crate's error type.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// A CUDA runtime, cuBLAS or kernel launch failure: the code and what was running.
    Cuda { code: i32, what: String },
    /// An argument or state the forward cannot take (wrong shape, too many rows, a slot in the
    /// wrong phase).
    Invalid(String),
    /// The device, the page pool or the slot pool has no room.
    OutOfMemory(String),
    /// Reading the checkpoint or the config failed.
    Model(glm53f_model::Error),
    /// Anything else, with context (an expert backend's failure, a host file).
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda { code, what } => write!(f, "{what}: CUDA error {code}"),
            Error::Invalid(m) => write!(f, "invalid: {m}"),
            Error::OutOfMemory(m) => write!(f, "out of memory: {m}"),
            Error::Model(e) => write!(f, "model: {e}"),
            Error::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<glm53f_model::Error> for Error {
    fn from(e: glm53f_model::Error) -> Self {
        Error::Model(e)
    }
}

impl From<Error> for String {
    fn from(e: Error) -> String {
        e.to_string()
    }
}

/// `Err(Error::Invalid)` with a formatted message.
macro_rules! invalid {
    ($($t:tt)*) => { $crate::error::Error::Invalid(format!($($t)*)) };
}
pub(crate) use invalid;
