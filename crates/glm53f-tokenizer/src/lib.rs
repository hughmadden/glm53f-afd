//! `glm53f-tokenizer`: GLM-5.3-Flash's tokenizer and chat template, std-only.
//!
//! - [`Tokenizer`]: the checkpoint's tokenizer.json, exactly: added tokens, the pre-tokenizer
//!   regex ([`pretok`], over the reference's own character classes, [`unicode`]),
//!   `ignore_merges` and byte-pair merging in the reference's order ([`bpe`]), and decode.
//! - [`stream`]: streaming detokenization that never splits a UTF-8 character and whose deltas
//!   concatenate to the whole decode.
//! - [`template`]: the chat template, rendered by hand.
//! - [`json`]: the JSON reader and the Python-compatible writer the loader and the template use.
//!
//! The stop tokens are `<|endoftext|>` (154,820), `<|user|>` (154,827) and `<|observation|>`
//! (154,829): [`STOP_TOKENS`], [`STOP_IDS`], [`Tokenizer::stop_ids`].
//!
//! ```no_run
//! use glm53f_tokenizer::{template, Tokenizer};
//! let tok = Tokenizer::from_file("tokenizer.json").unwrap();
//! let prompt = template::render(&[template::Message::new("user", "Hello")], &[], &template::Options::default()).unwrap();
//! let ids = tok.encode(&prompt);
//! let mut stream = glm53f_tokenizer::StreamDecoder::new(&tok, true);
//! let text: String = ids.iter().map(|&id| stream.push(id)).collect::<String>() + &stream.finish();
//! assert_eq!(text, tok.decode(&ids, true));
//! ```

pub mod bpe;
pub mod json;
pub mod pretok;
pub mod stream;
pub mod template;
pub mod tokenizer;
pub mod unicode;

pub use stream::{StreamDecoder, Utf8Stream};
pub use tokenizer::{AddedToken, Tokenizer, STOP_IDS, STOP_TOKENS};
