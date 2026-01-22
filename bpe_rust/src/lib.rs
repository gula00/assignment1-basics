use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use fancy_regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;

/// GPT-2 style pre-tokenization pattern
const GPT2_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pre-tokenize text using GPT-2 pattern
fn pretokenize(text: &str, pattern: &Regex) -> Vec<String> {
    pattern
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Compute word frequencies from text
fn compute_word_freqs(
    text: &str,
    special_tokens: &[String],
    num_special: usize,
    pattern: &Regex,
) -> FxHashMap<Vec<u32>, i64> {
    let mut word_freqs: FxHashMap<Vec<u32>, i64> = FxHashMap::default();

    // Split by special tokens using standard regex (no fancy features needed)
    let segments: Vec<&str> = if special_tokens.is_empty() {
        vec![text]
    } else {
        let special_pattern = special_tokens
            .iter()
            .map(|s| fancy_regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");
        let special_re = regex::Regex::new(&special_pattern).unwrap();
        special_re.split(text).collect()
    };

    for segment in segments {
        for word in pretokenize(segment, pattern) {
            let word_bytes = word.as_bytes();
            let word_ids: Vec<u32> = word_bytes
                .iter()
                .map(|&b| b as u32 + num_special as u32)
                .collect();
            *word_freqs.entry(word_ids).or_insert(0) += 1;
        }
    }

    word_freqs
}

/// Initialize vocabulary with special tokens and byte values
fn init_vocab(special_tokens: &[String]) -> (FxHashMap<u32, Vec<u8>>, u32) {
    let mut vocab: FxHashMap<u32, Vec<u8>> = FxHashMap::default();
    let mut idx: u32 = 0;

    // Add special tokens first
    for token in special_tokens {
        vocab.insert(idx, token.as_bytes().to_vec());
        idx += 1;
    }

    // Add all 256 byte values
    for i in 0..256u32 {
        vocab.insert(idx, vec![i as u8]);
        idx += 1;
    }

    (vocab, idx)
}

/// Perform BPE merge operations with optimized data structures
fn perform_merges(
    word_freqs: &mut FxHashMap<Vec<u32>, i64>,
    vocab: &mut FxHashMap<u32, Vec<u8>>,
    num_merges: usize,
    start_idx: u32,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = vocab.clone();
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(num_merges);
    let mut idx = start_idx;

    // Build pair counts and inverted index
    let mut pair_counts: FxHashMap<(u32, u32), i64> = FxHashMap::default();
    let mut pair_to_words: FxHashMap<(u32, u32), FxHashSet<Vec<u32>>> = FxHashMap::default();

    for (word, &freq) in word_freqs.iter() {
        if word.len() < 2 {
            continue;
        }
        for i in 0..word.len() - 1 {
            let pair = (word[i], word[i + 1]);
            *pair_counts.entry(pair).or_insert(0) += freq;
            pair_to_words.entry(pair).or_default().insert(word.clone());
        }
    }

    for _ in 0..num_merges {
        if pair_counts.is_empty() {
            break;
        }

        // Find best pair (max count, lexicographic tie-break)
        let best_pair = pair_counts
            .iter()
            .filter(|(_, &count)| count > 0)
            .max_by(|(p1, &c1), (p2, &c2)| {
                match c1.cmp(&c2) {
                    Ordering::Equal => {
                        let bytes1 = (&id_to_bytes[&p1.0], &id_to_bytes[&p1.1]);
                        let bytes2 = (&id_to_bytes[&p2.0], &id_to_bytes[&p2.1]);
                        bytes1.cmp(&bytes2)
                    }
                    other => other,
                }
            })
            .map(|(&pair, _)| pair);

        let best_pair = match best_pair {
            Some(p) => p,
            None => break,
        };

        let (a, b) = best_pair;
        let a_bytes = id_to_bytes[&a].clone();
        let b_bytes = id_to_bytes[&b].clone();

        // Create new token
        let mut new_bytes = a_bytes.clone();
        new_bytes.extend(&b_bytes);
        let new_id = idx;
        vocab.insert(new_id, new_bytes.clone());
        id_to_bytes.insert(new_id, new_bytes);
        merges.push((a_bytes, b_bytes));
        idx += 1;

        // Get affected words
        let affected_words: Vec<Vec<u32>> = pair_to_words
            .remove(&best_pair)
            .unwrap_or_default()
            .into_iter()
            .collect();
        pair_counts.remove(&best_pair);

        // Process affected words
        for word in affected_words {
            let freq = match word_freqs.remove(&word) {
                Some(f) => f,
                None => continue,
            };

            // Remove old pairs from counts and index
            for i in 0..word.len() - 1 {
                let p = (word[i], word[i + 1]);
                if let Some(count) = pair_counts.get_mut(&p) {
                    *count -= freq;
                }
                if let Some(words) = pair_to_words.get_mut(&p) {
                    words.remove(&word);
                }
            }

            // Merge the pair in the word
            let mut new_word: Vec<u32> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == a && word[i + 1] == b {
                    new_word.push(new_id);
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }

            // Update word_freqs
            *word_freqs.entry(new_word.clone()).or_insert(0) += freq;

            // Add new pairs
            for i in 0..new_word.len().saturating_sub(1) {
                let p = (new_word[i], new_word[i + 1]);
                *pair_counts.entry(p).or_insert(0) += freq;
                pair_to_words.entry(p).or_default().insert(new_word.clone());
            }
        }
    }

    merges
}

/// Train BPE tokenizer
#[pyfunction]
#[pyo3(signature = (input_path, vocab_size, special_tokens, merges_outpath=None, vocab_outpath=None))]
fn train_bpe<'py>(
    py: Python<'py>,
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
    merges_outpath: Option<&str>,
    vocab_outpath: Option<&str>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    // Compile regex pattern
    let pattern = Regex::new(GPT2_PATTERN).expect("Invalid regex pattern");

    // Read corpus
    let text = std::fs::read_to_string(input_path)?;

    // Pre-tokenize and compute word frequencies
    let num_special = special_tokens.len();
    let mut word_freqs = compute_word_freqs(&text, &special_tokens, num_special, &pattern);

    // Initialize vocabulary
    let (mut vocab, idx) = init_vocab(&special_tokens);

    // Calculate number of merges
    let num_merges = vocab_size.saturating_sub(vocab.len());

    let merges = if num_merges > 0 {
        perform_merges(&mut word_freqs, &mut vocab, num_merges, idx)
    } else {
        vec![]
    };

    // Save results if paths provided
    if let Some(path) = merges_outpath {
        let mut content = String::new();
        for (a, b) in &merges {
            content.push_str(&format!("{:?} {:?}\n", a, b));
        }
        std::fs::write(path, content)?;
    }

    if let Some(path) = vocab_outpath {
        let mut content = String::new();
        let mut items: Vec<_> = vocab.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        for (idx, bytes) in items {
            content.push_str(&format!("{}\t{:?}\n", idx, bytes));
        }
        std::fs::write(path, content)?;
    }

    // Convert to Python objects
    let py_vocab = PyDict::new_bound(py);
    for (idx, bytes) in &vocab {
        py_vocab.set_item(*idx, PyBytes::new_bound(py, bytes))?;
    }

    let py_merges = PyList::empty_bound(py);
    for (a, b) in &merges {
        let tuple = PyTuple::new_bound(py, &[
            PyBytes::new_bound(py, a).into_any(),
            PyBytes::new_bound(py, b).into_any(),
        ]);
        py_merges.append(tuple)?;
    }

    Ok((py_vocab, py_merges))
}

/// Python module
#[pymodule]
fn bpe_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}
