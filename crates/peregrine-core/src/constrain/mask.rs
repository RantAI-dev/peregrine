//! Token masks: a vocabulary's byte trie, and the allowed token ids for a matcher state.
//!
//! The model's vocabulary is turned into the bytes the decode loop appends to the reply
//! (`Tokenizer::token_bytes`), byte tries over them, and precomputed bitsets for the states
//! that repeat:
//!
//! * `plain`: every character is plain string content (valid UTF-8, none below `U+0020`,
//!   none `"` or `\`), by character count. Inside a string the mask is a clone of the
//!   cumulative bitset for the remaining `maxLength`, plus a walk of the small "boundary"
//!   trie (tokens that can close the string or write an escape).
//! * `quoted`: the same, for tokens that begin with `"` and continue with plain content.
//!   At a value or key position that is a plain string, the mask is a clone of that
//!   cumulative bitset plus the walk of the quoted boundary tokens, instead of walking
//!   every token that starts with a quote.
//! * everything else is walked through the full trie, with the branch pruned at the first
//!   byte the matcher rejects.
//!
//! Special/control tokens are never content; the end tokens are passed to
//! [`Vocabulary::allowed`] on every call and are added only when the matcher may end.

use super::matcher::{str_step, Matcher, StrStep};
use crate::tokenizer::Tokenizer;
use std::collections::HashMap;
use std::sync::Mutex;

/// Character counts are precomputed up to this many characters; longer plain tokens are
/// kept in a short side list and filtered one by one (they are rare).
const COUNT_CAP: u32 = 32;

/// One node of a byte trie. Children are sorted by byte for a deterministic walk.
struct Trie {
    children: Vec<Vec<(u8, u32)>>,
    /// The token ids whose bytes end exactly here (identical pieces share a node).
    tokens: Vec<Vec<u32>>,
}

impl Trie {
    fn build<I: Iterator<Item = (u32, Vec<u8>)>>(tokens: I) -> Trie {
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut leaves: Vec<Vec<u32>> = vec![Vec::new()];
        // A hash map of (parent, byte) → child while building: the linear scan of a
        // node's children is cheap per step but a 150k-token vocabulary makes it the
        // one-time preparation's dominant cost.
        let mut paths: HashMap<(u32, u8), u32> = HashMap::new();
        for (id, bytes) in tokens {
            let mut node = 0u32;
            for b in bytes {
                node = *paths.entry((node, b)).or_insert_with(|| {
                    let next = children.len() as u32;
                    children.push(Vec::new());
                    leaves.push(Vec::new());
                    next
                });
            }
            leaves[node as usize].push(id);
        }
        for ((parent, b), child) in paths {
            children[parent as usize].push((b, child));
        }
        for list in &mut children {
            list.sort_unstable_by_key(|&(b, _)| b);
        }
        Trie {
            children,
            tokens: leaves,
        }
    }
}

/// A set of token ids, stored as a bitset over the vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    bits: Vec<u64>,
    n: usize,
}

impl TokenMask {
    /// An empty mask for a vocabulary of `n` tokens.
    pub fn new(n: usize) -> TokenMask {
        TokenMask {
            bits: vec![0; n.div_ceil(64)],
            n,
        }
    }

    /// The vocabulary size.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    #[inline]
    pub(crate) fn insert(&mut self, id: u32) {
        let i = id as usize;
        if i < self.n {
            self.bits[i >> 6] |= 1u64 << (i & 63);
        }
    }

    #[inline]
    pub(crate) fn remove(&mut self, id: u32) {
        let i = id as usize;
        if i < self.n {
            self.bits[i >> 6] &= !(1u64 << (i & 63));
        }
    }

    /// Whether `id` is allowed.
    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        let i = id as usize;
        i < self.n && self.bits[i >> 6] & (1u64 << (i & 63)) != 0
    }

    /// The set bits, ascending.
    pub fn iter(&self) -> TokenMaskIter<'_> {
        TokenMaskIter {
            bits: &self.bits,
            word: 0,
            cur: 0,
        }
    }

    /// The allowed ids, ascending.
    pub fn ids(&self) -> Vec<u32> {
        self.iter().collect()
    }

    /// How many tokens are allowed.
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn insert_all(&mut self, other: &TokenMask) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= *b;
        }
    }

    fn intersect(&mut self, other: &TokenMask) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a &= *b;
        }
    }

    fn without(&mut self, other: &TokenMask) {
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a &= !*b;
        }
    }
}

/// Ascending iterator over a [`TokenMask`].
pub struct TokenMaskIter<'a> {
    bits: &'a [u64],
    word: usize,
    cur: u64,
}

impl Iterator for TokenMaskIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        loop {
            if self.cur != 0 {
                let bit = self.cur.trailing_zeros();
                self.cur &= self.cur - 1;
                return Some((self.word - 1) as u32 * 64 + bit);
            }
            if self.word >= self.bits.len() {
                return None;
            }
            self.cur = self.bits[self.word];
            self.word += 1;
        }
    }
}

/// A model's vocabulary as the mask needs it.
pub struct Vocabulary {
    n_vocab: usize,
    /// Raw bytes per token id (empty for a token that is never content).
    bytes: Vec<Vec<u8>>,
    content: Trie,
    quoted: Trie,
    /// Content-boundary tokens that close a string, with the characters before the close
    /// and the bytes after it.
    closing: Vec<ClosingToken>,
    closing_by_chars: Vec<TokenMask>,
    /// The closing tokens whose remainder the stack below the string accepts, by that
    /// stack. A closing token's validity after the quote depends on the parent, not on the
    /// string's character count.
    closing_parents: Mutex<HashMap<u64, (Vec<super::matcher::Item>, TokenMask)>>,
    /// Cumulative `plain` masks: index `c` holds the tokens with a plain character count
    /// at most `c` (`c < COUNT_CAP`); longer ones are in `long_plain`.
    plain_by_count: Vec<TokenMask>,
    long_plain: Vec<(u32, u32)>,
    /// Boundary tokens that do not close the string, by the characters they add (an
    /// escape or a pending UTF-8 sequence counts as one). A bounded `maxLength` is then a
    /// bitset index, not a fresh trie walk.
    nonclosing_by_need: Vec<TokenMask>,
    long_nonclosing: Vec<(u32, u32)>,
    /// The same, for tokens that begin with `"` and continue with plain content.
    quoted_by_count: Vec<TokenMask>,
    long_quoted: Vec<(u32, u32)>,
    /// The boundary walks by canonical string state: the tokens that close the string,
    /// write an escape, or end in the middle of one. They depend on the character count
    /// only through the string's min/max categories, so a long string pays for one walk.
    boundary_cache: Mutex<HashMap<u64, (Matcher, TokenMask, u64)>>,
    boundary_tick: std::sync::atomic::AtomicU64,
    /// Whole masks by (state, end tokens), shared by every slot of this model. A repeated
    /// structural state — the inside of a string, the position after a comma — is a hit
    /// after its first request.
    mask_cache: Mutex<MaskMemo>,
    mask_tick: std::sync::atomic::AtomicU64,
}

type MaskEntry = (Matcher, Vec<u32>, TokenMask, u64);

#[derive(Clone)]
struct ClosingToken {
    id: u32,
    chars: u32,
    remainder: Vec<u8>,
}

#[derive(Default)]
struct MaskMemo {
    entries: HashMap<u64, Vec<MaskEntry>>,
    len: usize,
}

impl Vocabulary {
    /// Build from a model's tokenizer: detokenised bytes, special tokens excluded.
    pub fn new(tk: &Tokenizer) -> Vocabulary {
        let n = tk.n_vocab;
        let mut special = vec![false; n];
        for id in tk.special_ids() {
            if (id as usize) < n {
                special[id as usize] = true;
            }
        }
        let mut bytes = vec![Vec::new(); n];
        let mut content = Vec::with_capacity(n);
        for id in 0..n as u32 {
            if special[id as usize] {
                continue;
            }
            let b = tk.token_bytes(id);
            if b.is_empty() {
                continue;
            }
            bytes[id as usize] = b.clone();
            content.push((id, b));
        }
        Self::from_content(n, content)
    }

    /// Build from raw token bytes (tests, and callers with their own tokenizer). A token
    /// with empty bytes is never content; end tokens may be passed to
    /// [`allowed`](Self::allowed) and are filtered out of the content mask there.
    pub fn from_tokens(tokens: Vec<Vec<u8>>) -> Vocabulary {
        let n = tokens.len();
        let content: Vec<(u32, Vec<u8>)> = tokens
            .into_iter()
            .enumerate()
            .filter(|(_, b)| !b.is_empty())
            .map(|(i, b)| (i as u32, b))
            .collect();
        Self::from_content(n, content)
    }

    fn from_content(n: usize, content: Vec<(u32, Vec<u8>)>) -> Vocabulary {
        let mut bytes = vec![Vec::new(); n];
        let mut plain = Vec::new();
        let mut closing: Vec<ClosingToken> = Vec::new();
        let mut nonclosing = Vec::new();
        let mut quoted_plain = Vec::new();
        let mut quoted_boundary = Vec::new();
        for (id, b) in &content {
            bytes[*id as usize] = b.clone();
            let class = plain_or_quoted(b);
            match class {
                Some((count, true)) => quoted_plain.push((*id, count)),
                Some((count, false)) => plain.push((*id, count)),
                None if b.first() == Some(&b'"') => quoted_boundary.push((*id, b[1..].to_vec())),
                None => {}
            }
            // The content boundary: tokens that are not plain content. A token that
            // closes the string needs the parent's state after the quote; one that does
            // not is allowed whenever its characters still fit the remaining length.
            if !matches!(class, Some((_, false))) {
                match class_boundary(b) {
                    Boundary::Closing { chars, remainder } => closing.push(ClosingToken {
                        id: *id,
                        chars,
                        remainder,
                    }),
                    Boundary::NonClosing(need) => nonclosing.push((*id, need)),
                    Boundary::Invalid => {}
                }
            }
        }
        let mut plain_by_count = Vec::new();
        let mut long_plain = Vec::new();
        let mut nonclosing_by_need = Vec::new();
        let mut long_nonclosing = Vec::new();
        let mut quoted_by_count = Vec::new();
        let mut long_quoted = Vec::new();
        build_counts(n, &plain, &mut plain_by_count, &mut long_plain);
        build_counts(
            n,
            &nonclosing,
            &mut nonclosing_by_need,
            &mut long_nonclosing,
        );
        build_counts(n, &quoted_plain, &mut quoted_by_count, &mut long_quoted);
        let quoted_trie = Trie::build(quoted_boundary.into_iter());
        // Cumulative `closing_by_chars[k]`: closing tokens with at most k characters
        // before the quote, for k up to COUNT_CAP; the rest are filtered one by one.
        let mut closing_by_chars = Vec::new();
        let mut mask = TokenMask::new(n);
        for k in 0..=closing
            .iter()
            .map(|c| c.chars.min(COUNT_CAP))
            .max()
            .unwrap_or(0)
        {
            for ct in &closing {
                if ct.chars.min(COUNT_CAP) == k && ct.chars <= COUNT_CAP {
                    mask.insert(ct.id);
                }
            }
            closing_by_chars.push(mask.clone());
        }
        // The full trie is every token, including the quoted ones: the generic walk needs
        // them when the quoted fast path does not apply, and an object-key or enum literal
        // can expect any byte in its middle.
        let all: Vec<(u32, Vec<u8>)> = content;
        let content_trie = Trie::build(all.into_iter());
        Vocabulary {
            n_vocab: n,
            bytes,
            content: content_trie,
            quoted: quoted_trie,
            closing,
            closing_by_chars,
            closing_parents: Mutex::new(HashMap::new()),
            plain_by_count,
            long_plain,
            nonclosing_by_need,
            long_nonclosing,
            quoted_by_count,
            long_quoted,
            boundary_cache: Mutex::new(HashMap::new()),
            boundary_tick: std::sync::atomic::AtomicU64::new(0),
            mask_cache: Mutex::new(MaskMemo::default()),
            mask_tick: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The vocabulary's size.
    pub fn len(&self) -> usize {
        self.n_vocab
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.n_vocab == 0
    }

    /// The bytes the engine appends to the reply for `id` (empty when it is not content).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// The token ids that keep the reply a valid prefix of the constraint.
    ///
    /// `ends` are the slot's stop tokens: they are allowed only when the matcher may end,
    /// and never as content. Masks are memoised by (state, ends) for the whole model, so a
    /// repeated structural state is a hit after its first request.
    pub fn allowed(&self, matcher: &Matcher, ends: &[u32]) -> TokenMask {
        let key = matcher.state_hash() ^ hash_ends(ends);
        {
            let cache = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(list) = cache.entries.get(&key) {
                for (state, state_ends, mask, _) in list {
                    if state == matcher && state_ends == ends {
                        return mask.clone();
                    }
                }
            }
        }
        let mask = self.compute_allowed(matcher, ends);
        let tick = self
            .mask_tick
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut cache = self.mask_cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len >= 512 {
            let mut oldest: Option<(u64, usize, u64)> = None;
            for (&k, list) in cache.entries.iter() {
                for (i, (_, _, _, t)) in list.iter().enumerate() {
                    if oldest.as_ref().is_none_or(|&(_, _, ot)| *t < ot) {
                        oldest = Some((k, i, *t));
                    }
                }
            }
            if let Some((k, i, _)) = oldest {
                if let Some(list) = cache.entries.get_mut(&k) {
                    list.remove(i);
                    if list.is_empty() {
                        cache.entries.remove(&k);
                    }
                }
                cache.len -= 1;
            }
        }
        cache.entries.entry(key).or_default().push((
            matcher.clone(),
            ends.to_vec(),
            mask.clone(),
            tick,
        ));
        cache.len += 1;
        mask
    }

    fn compute_allowed(&self, matcher: &Matcher, ends: &[u32]) -> TokenMask {
        let mut mask = TokenMask::new(self.n_vocab);
        if matcher.is_closed() {
            for &e in ends {
                mask.insert(e);
            }
            return mask;
        }
        if let Some((chars, _min, max)) = matcher.simple_string() {
            self.string_content(&mut mask, matcher, chars, max);
        } else {
            // A value or key position: a token that begins with `"` and continues with
            // plain content is allowed through the precomputed cumulative bitset when the
            // state just after the quote is a plain string.
            let mut skip_quote = false;
            let mut after_quote = matcher.clone();
            if matcher.accepts(b'"') && after_quote.advance(b'"') {
                if let Some((chars, _min, max)) = after_quote.simple_string() {
                    skip_quote = true;
                    self.quoted_content(&mut mask, chars, max);
                    let boundary = self.boundary_mask(&after_quote, true);
                    mask.insert_all(&boundary);
                }
            }
            let skip = skip_quote.then_some(b'"');
            self.walk(&self.content, matcher, 0, &mut mask, skip.unwrap_or(b'\0'));
        }
        for &e in ends {
            mask.remove(e);
        }
        if matcher.can_end() {
            for &e in ends {
                mask.insert(e);
            }
        }
        mask
    }

    fn string_content(&self, mask: &mut TokenMask, matcher: &Matcher, chars: u32, max: u32) {
        let remaining = max.saturating_sub(chars);
        self.plain_bits(mask, remaining);
        self.nonclosing_bits(mask, remaining);
        if let Some((below, _)) = matcher.string_split() {
            let (min, max) = matcher.string_bounds();
            self.closing_bits(mask, matcher, below, chars, min, max);
        }
    }

    /// The closing tokens that fit `chars` characters and whose bytes after the quote the
    /// stack below the string accepts.
    fn closing_bits(
        &self,
        mask: &mut TokenMask,
        matcher: &Matcher,
        below: &[super::matcher::Item],
        chars: u32,
        min: u32,
        max: u32,
    ) {
        let lo = min.saturating_sub(chars);
        let hi = max.saturating_sub(chars);
        if lo > hi {
            return;
        }
        let mut allowed = TokenMask::new(self.n_vocab);
        if !self.closing_by_chars.is_empty() {
            let hi_index = (hi.min(COUNT_CAP) as usize).min(self.closing_by_chars.len() - 1);
            allowed.insert_all(&self.closing_by_chars[hi_index]);
            if lo > 0 {
                let lo_index =
                    ((lo - 1).min(COUNT_CAP) as usize).min(self.closing_by_chars.len() - 1);
                allowed.without(&self.closing_by_chars[lo_index]);
            }
        }
        // Closing tokens longer than COUNT_CAP are rare; test them directly.
        for ct in &self.closing {
            if ct.chars > COUNT_CAP && ct.chars >= lo && ct.chars <= hi {
                allowed.insert(ct.id);
            }
        }
        let parent = self.closing_parent_bits(matcher, below);
        allowed.intersect(&parent);
        mask.insert_all(&allowed);
    }

    fn closing_parent_bits(&self, matcher: &Matcher, below: &[super::matcher::Item]) -> TokenMask {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let key = {
            let mut h = DefaultHasher::new();
            below.hash(&mut h);
            h.finish()
        };
        {
            let cache = self
                .closing_parents
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some((state, mask)) = cache.get(&key) {
                if state.as_slice() == below {
                    return mask.clone();
                }
            }
        }
        let base = matcher.continuing_below(below);
        let allowed = base.allowed_bytes();
        let mut mask = TokenMask::new(self.n_vocab);
        for ct in &self.closing {
            // Most closing tokens' remainder starts with a byte the parent does not
            // allow; a bit test avoids cloning the matcher for those.
            if let Some(&first) = ct.remainder.first() {
                if !allowed.contains(first) {
                    continue;
                }
            }
            let mut next = base.clone();
            if next.advance_bytes(&ct.remainder).is_ok() {
                mask.insert(ct.id);
            }
        }
        let mut cache = self
            .closing_parents
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(key, (below.to_vec(), mask.clone()));
        mask
    }

    /// The boundary tokens for a canonical string state, computed once per category and
    /// cached. A long string revisits the same key for every character.
    fn boundary_mask(&self, matcher: &Matcher, quoted: bool) -> TokenMask {
        let canonical = matcher
            .canonical_string()
            .unwrap_or_else(|| matcher.clone());
        let key =
            canonical
                .state_hash()
                .wrapping_add(if quoted { 0x9E37_79B9_7F4A_7C15 } else { 0 });
        {
            let cache = self
                .boundary_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some((state, mask, _)) = cache.get(&key) {
                if state == &canonical {
                    return mask.clone();
                }
            }
        }
        let mut mask = TokenMask::new(self.n_vocab);
        if quoted {
            self.walk_string(&self.quoted, &canonical, 0, &mut mask, b'\0');
        }
        let tick = self
            .boundary_tick
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut cache = self
            .boundary_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache.len() >= 128 {
            if let Some((&old, _)) = cache.iter().min_by_key(|(_, (_, _, t))| *t) {
                cache.remove(&old);
            }
        }
        cache.insert(key, (canonical, mask.clone(), tick));
        mask
    }

    fn nonclosing_bits(&self, mask: &mut TokenMask, remaining: u32) {
        if !self.nonclosing_by_need.is_empty() {
            let index = (remaining.min(COUNT_CAP) as usize).min(self.nonclosing_by_need.len() - 1);
            mask.insert_all(&self.nonclosing_by_need[index]);
        }
        for &(id, need) in &self.long_nonclosing {
            if need <= remaining {
                mask.insert(id);
            }
        }
    }

    fn quoted_content(&self, mask: &mut TokenMask, chars: u32, max: u32) {
        let remaining = max.saturating_sub(chars);
        self.quoted_bits(mask, remaining);
    }

    fn plain_bits(&self, mask: &mut TokenMask, remaining: u32) {
        let index = (remaining.min(COUNT_CAP) as usize).min(self.plain_by_count.len() - 1);
        if !self.plain_by_count.is_empty() {
            mask.insert_all(&self.plain_by_count[index]);
        }
        for &(id, count) in &self.long_plain {
            if count <= remaining {
                mask.insert(id);
            }
        }
    }

    fn quoted_bits(&self, mask: &mut TokenMask, remaining: u32) {
        if !self.quoted_by_count.is_empty() {
            let index = (remaining.min(COUNT_CAP) as usize).min(self.quoted_by_count.len() - 1);
            mask.insert_all(&self.quoted_by_count[index]);
        }
        for &(id, count) in &self.long_quoted {
            if count <= remaining {
                mask.insert(id);
            }
        }
    }

    /// Walk a token trie inside a string with the string frame alone: the trie's nodes
    /// then cost a `Copy` state transition, not a matcher clone. A token that closes the
    /// string hands the rest of its bytes to the general walk from the stack below.
    fn walk_string(
        &self,
        trie: &Trie,
        matcher: &Matcher,
        node: u32,
        mask: &mut TokenMask,
        skip: u8,
    ) {
        let Some((below, item)) = matcher.string_split() else {
            self.walk(trie, matcher, node, mask, skip);
            return;
        };
        let schema = matcher.string_schema(item);
        self.walk_string_node(trie, matcher, below, schema, item, node, mask, skip);
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_string_node(
        &self,
        trie: &Trie,
        matcher: &Matcher,
        below: &[super::matcher::Item],
        schema: super::schema::StringSchema,
        item: super::matcher::StrItem,
        node: u32,
        mask: &mut TokenMask,
        skip: u8,
    ) {
        for &id in &trie.tokens[node as usize] {
            mask.insert(id);
        }
        for &(b, child) in &trie.children[node as usize] {
            if node == 0 && skip != 0 && b == skip {
                continue;
            }
            match str_step(schema, item, b) {
                StrStep::More(next) => {
                    self.walk_string_node(trie, matcher, below, schema, next, child, mask, 0)
                }
                StrStep::Done => {
                    let next = matcher.continuing_below(below);
                    self.walk(trie, &next, child, mask, 0);
                }
                StrStep::No => {}
            }
        }
    }

    /// Walk `trie` with `matcher`, inserting every leaf whose bytes leave a non-empty
    /// state. At the root, `skip` (when not zero) is not descended into.
    fn walk(&self, trie: &Trie, matcher: &Matcher, node: u32, mask: &mut TokenMask, skip: u8) {
        for &id in &trie.tokens[node as usize] {
            mask.insert(id);
        }
        // Check the byte before cloning the state: a branch whose first byte the
        // matcher rejects costs one bit test, not a clone and a dead advance.
        let allowed = matcher.allowed_bytes();
        for &(b, child) in &trie.children[node as usize] {
            if node == 0 && skip != 0 && b == skip {
                continue;
            }
            if !allowed.contains(b) {
                continue;
            }
            let mut next = matcher.clone();
            if next.advance(b) {
                // A token may write whitespace and then open a string. Once the walk is
                // inside a string, use the allocation-free string transitions: the
                // generic path would clone the matcher at every trie node of the
                // string's subtree.
                if let Some((below, item)) = next.string_split() {
                    let schema = next.string_schema(item);
                    self.walk_string_node(trie, &next, below, schema, item, child, mask, 0);
                } else {
                    self.walk(trie, &next, child, mask, b'\0');
                }
            }
        }
    }
}

enum Boundary {
    Closing { chars: u32, remainder: Vec<u8> },
    NonClosing(u32),
    Invalid,
}

/// How a non-plain token behaves from a plain string's between-characters state: it
/// closes the string, adds characters without closing, or cannot be written there.
fn class_boundary(bytes: &[u8]) -> Boundary {
    let schema = super::schema::StringSchema {
        min: 0,
        max: u32::MAX,
    };
    let mut item = super::matcher::StrItem {
        string: 0,
        state: super::matcher::StrState::Plain,
        chars: 0,
    };
    for (i, &b) in bytes.iter().enumerate() {
        match str_step(schema, item, b) {
            StrStep::More(next) => item = next,
            StrStep::Done => {
                return Boundary::Closing {
                    chars: item.chars,
                    remainder: bytes[i + 1..].to_vec(),
                }
            }
            StrStep::No => return Boundary::Invalid,
        }
    }
    let pending = if item.state == super::matcher::StrState::Plain {
        0
    } else {
        1
    };
    Boundary::NonClosing(item.chars + pending)
}

/// `Some((characters, begins_with_quote))` for a token that is plain string content (or a
/// `"` followed by plain content); `None` when it closes/escapes or is not valid UTF-8.
fn plain_or_quoted(bytes: &[u8]) -> Option<(u32, bool)> {
    let (quoted, body) = match bytes.first() {
        Some(b'"') => (true, &bytes[1..]),
        _ => (false, bytes),
    };
    let text = std::str::from_utf8(body).ok()?;
    let mut n = 0u32;
    for c in text.chars() {
        let u = c as u32;
        if u < 0x20 || c == '"' || c == '\\' {
            return None;
        }
        n += 1;
    }
    Some((n, quoted))
}

fn hash_ends(ends: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &e in ends {
        h ^= e as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn build_counts(
    n: usize,
    items: &[(u32, u32)],
    by_count: &mut Vec<TokenMask>,
    long: &mut Vec<(u32, u32)>,
) {
    let max = items
        .iter()
        .map(|&(_, c)| c.min(COUNT_CAP))
        .max()
        .unwrap_or(0);
    let mut mask = TokenMask::new(n);
    for c in 0..=max {
        for &(id, count) in items {
            if count.min(COUNT_CAP) == c && count <= COUNT_CAP {
                mask.insert(id);
            }
        }
        by_count.push(mask.clone());
    }
    for &(id, count) in items {
        if count > COUNT_CAP {
            long.push((id, count));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrain::{Constraint, Schema};
    use crate::json::Value as J;

    fn vocab() -> Vocabulary {
        let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        tokens.push(b"true".to_vec());
        tokens.push(b"false".to_vec());
        tokens.push(b"\"abc\"".to_vec());
        tokens.push(b"{\"".to_vec());
        tokens.push(b"\": ".to_vec());
        tokens.push(b",\"".to_vec());
        Vocabulary::from_tokens(tokens)
    }

    fn schema(text: &str) -> Schema {
        Schema::new(J::parse(text).unwrap())
    }

    #[test]
    fn the_mask_is_the_brute_force_mask() {
        let v = vocab();
        let c = Constraint::from_schema(schema(
            r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"],"additionalProperties":false}"#,
        ));
        let mut m = Matcher::new(&c).unwrap();
        let ends = [300u32];
        let prefixes: [&[u8]; 4] = [b"", b"{\"", b"{\"a\": 1", b"{\"a\": 1, \"b\": \"x"];
        for prefix in prefixes {
            let mut state = m.clone();
            assert!(state.advance_bytes(prefix).is_ok(), "{prefix:?}");
            let mut want = Vec::new();
            for id in 0..v.len() as u32 {
                let bytes = v.token_bytes(id);
                if bytes.is_empty() || ends.contains(&id) {
                    continue;
                }
                let mut next = state.clone();
                if next.advance_bytes(bytes).is_ok() {
                    want.push(id);
                }
            }
            if state.can_end() {
                want.extend(ends.iter().copied().filter(|&e| (e as usize) < v.len()));
            }
            want.sort_unstable();
            want.dedup();
            let got = v.allowed(&state, &ends).ids();
            assert_eq!(got, want, "after {prefix:?}");
        }
        let _ = &mut m;
    }

    #[test]
    fn a_closed_value_allows_only_end_tokens() {
        let v = vocab();
        let c = Constraint::from_schema(schema(r#"{"type":"array","maxItems":1}"#));
        let mut m = Matcher::new(&c).unwrap();
        assert!(m.advance_bytes(b"[]").is_ok());
        assert!(m.is_closed());
        assert_eq!(v.allowed(&m, &[261]).ids(), vec![261]);
    }
}
