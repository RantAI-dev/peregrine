//! Stop-sequence matching for `/v1/chat/completions`.
//!
//! A stop string may be split across token boundaries, so it cannot be matched one token at a time: the
//! matcher keeps the longest suffix of the text seen so far that is still a prefix of some stop sequence
//! (a *partial match*) and emits everything before it. Whether it is streamed or collected, the caller
//! therefore never sees a character that belongs to a stop sequence.

/// One request's stop sequences and the text not yet known to be free of them.
pub(crate) struct StopMatcher {
    sequences: Vec<String>,
    /// Text held back because its suffix may be the beginning of a stop sequence.
    pending: String,
    stopped: bool,
}

impl StopMatcher {
    pub(crate) fn new(sequences: Vec<String>) -> StopMatcher {
        StopMatcher {
            sequences,
            pending: String::new(),
            stopped: false,
        }
    }

    /// Feed a decoded piece of text and get back what is safe to emit now.
    ///
    /// The second value is `true` once a stop sequence was found. The returned text is then everything
    /// before the sequence; the sequence itself is dropped, as is anything after it.
    pub(crate) fn feed(&mut self, text: &str) -> (String, bool) {
        if self.sequences.is_empty() || self.stopped || text.is_empty() {
            return (text.to_string(), self.stopped);
        }
        self.pending.push_str(text);
        if let Some(at) = self.earliest_match() {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            self.stopped = true;
            return (out, true);
        }
        let hold = self.partial_suffix_len();
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    /// The text still held back, for the end of a generation that stopped without a stop sequence.
    pub(crate) fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Byte offset of the earliest stop sequence in `pending`.
    fn earliest_match(&self) -> Option<usize> {
        self.sequences
            .iter()
            .filter_map(|s| self.pending.find(s))
            .min()
    }

    /// Length in bytes of the longest suffix of `pending` that starts a stop sequence.
    ///
    /// A suffix that is *all* of `pending` is held too: the sequence may be completed by the next piece.
    /// Only whole characters are considered, so a multi-byte code point is never split.
    fn partial_suffix_len(&self) -> usize {
        for (i, _) in self.pending.char_indices() {
            let suffix = &self.pending[i..];
            if self.sequences.iter().any(|s| s.starts_with(suffix)) {
                // Ascending order: the first matching suffix is the longest one.
                return suffix.len();
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::StopMatcher;

    /// Feed `pieces` one at a time and collect the emitted text and the stop flag.
    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for piece in pieces {
            let (text, hit) = m.feed(piece);
            out.push_str(&text);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&m.flush());
        (out, false)
    }

    #[test]
    fn without_stop_sequences_text_passes_through() {
        let mut m = StopMatcher::new(vec![]);
        assert_eq!(m.feed("hello "), ("hello ".into(), false));
        assert_eq!(m.feed("world"), ("world".into(), false));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn a_stop_sequence_is_dropped_and_truncates_the_reply() {
        let (text, hit) = run(&["\n\n"], &["hello", "\n", "\n", "ignored"]);
        assert!(hit);
        assert_eq!(text, "hello");
    }

    #[test]
    fn a_partial_match_is_held_across_token_boundaries() {
        let mut m = StopMatcher::new(vec!["</s>".into()]);
        for piece in ["<", "/", "s"] {
            assert_eq!(m.feed(piece), (String::new(), false), "held {piece}");
        }
        assert_eq!(m.feed(">tail"), (String::new(), true));
        assert_eq!(m.flush(), "");
    }

    #[test]
    fn a_failed_partial_match_is_emitted_whole() {
        // "ab" could start "abd", but "c" rules it out; nothing may be lost or duplicated.
        let (text, hit) = run(&["abd"], &["ab", "c"]);
        assert!(!hit);
        assert_eq!(text, "abc");
    }

    #[test]
    fn the_earliest_and_overlapping_candidates_win() {
        // "bc" starts one character before "abc" would, so "x" is the whole reply.
        let (text, hit) = run(&["abc", "bc"], &["xb", "c", "abc"]);
        assert!(hit);
        assert_eq!(text, "x");

        // The same start, two candidates: either drops the same suffix.
        let (text, hit) = run(&["abcd", "ab"], &["xab", "cd"]);
        assert!(hit);
        assert_eq!(text, "x");
    }

    #[test]
    fn a_stop_sequence_is_found_across_several_pieces() {
        let (text, hit) = run(&["END"], &["one", " two ", "E", "N", "D", " three"]);
        assert!(hit);
        assert_eq!(text, "one two ");
    }

    #[test]
    fn utf8_text_and_sequences_are_never_split() {
        // A multi-byte stop sequence, fed a character at a time: nothing is emitted early.
        let mut m = StopMatcher::new(vec!["é🙂".into()]);
        assert_eq!(m.feed("caf"), ("caf".into(), false));
        assert_eq!(m.feed("é"), (String::new(), false));
        assert_eq!(m.feed("🙂"), (String::new(), true));
        assert_eq!(m.flush(), "");

        // A failed partial match flushes the whole multi-byte character.
        let (text, hit) = run(&["éx"], &["café", "!"]);
        assert!(!hit);
        assert_eq!(text, "café!");
    }

    #[test]
    fn text_after_a_stop_sequence_is_not_emitted() {
        let (text, hit) = run(&["STOP"], &["before", "STOP", "after"]);
        assert!(hit);
        assert_eq!(text, "before");
    }

    #[test]
    fn an_empty_stop_sequence_stops_without_output() {
        let (text, hit) = run(&[""], &["anything"]);
        assert!(hit);
        assert_eq!(text, "");
    }

    #[test]
    fn flush_returns_a_held_partial_match() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.feed("the EN"), ("the ".into(), false));
        assert_eq!(m.flush(), "EN");
    }
}
