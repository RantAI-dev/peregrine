//! The incremental byte-level recogniser.
//!
//! The compiled grammar is a nondeterministic pushdown automaton. A state is a *set of
//! stacks* of [`Item`]s: each stack is one way the prefix can still be parsed, and the set
//! is kept sorted and deduplicated. The top of a stack is the active instruction; the items
//! below it are the suspended parents.
//!
//! * [`Matcher::allowed_bytes`] answers "which bytes may come next" (the union over the
//!   stacks, including the byte a skippable number or repetition may not consume).
//! * [`Matcher::advance`] feeds one byte: every stack that can consume it branches, the
//!   branches are closed (epsilon expansions), and the dead ones drop out.
//! * [`Matcher::can_end`] answers "may the reply end here": every frame left is one that
//!   may finish (an accepting number, a repetition at or past its minimum).
//! * [`Matcher::is_closed`] is stronger: the top-level value closed, so no content byte is
//!   possible and only an end token may follow.
//!
//! The state is all scalar `Copy` items, so cloning and hashing it are cheap.

use super::schema::{utf8_lead_bytes, Additional, ByteSet, Compiled, Node, Sign, UntilSchema};
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The most stacks one state may grow to before the schema is called too complex to serve.
/// A pathological `anyOf`/optional-property schema can branch; this bounds the work instead
/// of letting one request run the machine out of memory.
const MAX_STACKS: usize = 4096;

/// The work budget of one closure step, in item expansions. Far above any real schema.
const MAX_EXPANSIONS: usize = 200_000;

/// Bounded JSON whitespace: nothing, one space, or one or two newlines and at most twenty
/// spaces/tabs (llama.cpp v0.9.0's `SPACE_RULE`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum Ws {
    Start,
    Space,
    Nl(u8),
    NlSpace(u8, u8),
}

impl Ws {
    fn allowed(self) -> ByteSet {
        let mut set = ByteSet::EMPTY;
        match self {
            Ws::Start => {
                set.insert(b' ');
                set.insert(b'\n');
            }
            Ws::Space => {}
            Ws::Nl(n) => {
                if n < 2 {
                    set.insert(b'\n');
                }
                set.insert(b' ');
                set.insert(b'\t');
            }
            Ws::NlSpace(_, spaces) => {
                if spaces < 20 {
                    set.insert(b' ');
                    set.insert(b'\t');
                }
            }
        }
        set
    }

    fn step(self, b: u8) -> Option<Ws> {
        match self {
            Ws::Start if b == b' ' => Some(Ws::Space),
            Ws::Start if b == b'\n' => Some(Ws::Nl(1)),
            Ws::Nl(n) if b == b'\n' && n < 2 => Some(Ws::Nl(n + 1)),
            Ws::Nl(n) if b == b' ' || b == b'\t' => Some(Ws::NlSpace(n, 1)),
            Ws::NlSpace(n, spaces) if (b == b' ' || b == b'\t') && spaces < 20 => {
                Some(Ws::NlSpace(n, spaces + 1))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum ObjStage {
    /// The opening `{` has not been read yet.
    Open,
    /// After `{`: a key, or `}` when every remaining property is optional.
    KeyOrEnd,
    /// After `,`: a key is required (a trailing comma is not JSON).
    AfterComma,
    /// The key is in: a colon follows.
    AfterKey(u32),
    /// A colon is in: the property's value follows.
    BeforeValue(u32),
    /// A value is in: a comma or `}` follows.
    AfterValue,
    /// An additional property's key is in: a colon follows.
    AfterAddKey,
    /// The colon is in: an additional property's value follows.
    BeforeAddValue,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct ObjItem {
    pub(crate) object: u32,
    /// Index of the next declared property not yet emitted or skipped.
    pub(crate) next: u32,
    pub(crate) stage: ObjStage,
    pub(crate) ws: Ws,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum ArrStage {
    Open,
    FirstOrEnd,
    AfterValue,
    AfterComma,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct ArrItem {
    pub(crate) array: u32,
    pub(crate) index: u32,
    pub(crate) stage: ArrStage,
    pub(crate) ws: Ws,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum StrState {
    /// The opening quote has not been read yet.
    Start,
    /// Between characters.
    Plain,
    /// After a backslash.
    Escape,
    /// In `\uXXXX`.
    EscapeU { digits: u8, value: u16 },
    /// A high surrogate was read: a backslash follows.
    HighSlash,
    /// The backslash is in: a `u` follows.
    HighU,
    /// The low surrogate's four hex digits.
    LowU { digits: u8, value: u16 },
    /// Inside a raw multi-byte UTF-8 sequence: `lo..=hi` is the range of the next byte.
    Utf8 { need: u8, lo: u8, hi: u8 },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct StrItem {
    pub(crate) string: u32,
    pub(crate) state: StrState,
    /// Characters completed so far (an escape is one; a surrogate pair is one).
    pub(crate) chars: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum NumState {
    Start,
    /// After `-`.
    Minus,
    /// After `0`.
    Zero,
    /// After `[1-9][0-9]*`.
    Int,
    /// After `.`.
    Dot,
    /// After at least one fractional digit.
    Frac,
    /// After `e`/`E`.
    Exp,
    /// After an exponent sign.
    ExpSign,
    /// After at least one exponent digit.
    ExpDigits,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct NumItem {
    pub(crate) number: u32,
    pub(crate) state: NumState,
}

/// Where a free-text part stands inside the UTF-8 character it is reading: `need`
/// continuation bytes are still owed (`0` = between characters), and the next one must
/// lie in `lo..=hi` (the first continuation byte's range depends on the lead byte, which
/// is how overlong forms, surrogates and code points above `U+10FFFF` are refused).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct Utf8Tail {
    need: u8,
    lo: u8,
    hi: u8,
}

impl Utf8Tail {
    const BOUNDARY: Utf8Tail = Utf8Tail {
        need: 0,
        lo: 0,
        hi: 0,
    };

    #[inline]
    fn step(self, b: u8) -> Option<Utf8Tail> {
        let tail = |need, lo, hi| Some(Utf8Tail { need, lo, hi });
        if self.need == 0 {
            return match b {
                0x00..=0x7F => Some(Utf8Tail::BOUNDARY),
                0xC2..=0xDF => tail(1, 0x80, 0xBF),
                0xE0 => tail(2, 0xA0, 0xBF),
                0xED => tail(2, 0x80, 0x9F),
                0xE1..=0xEC | 0xEE..=0xEF => tail(2, 0x80, 0xBF),
                0xF0 => tail(3, 0x90, 0xBF),
                0xF1..=0xF3 => tail(3, 0x80, 0xBF),
                0xF4 => tail(3, 0x80, 0x8F),
                _ => None,
            };
        }
        if b < self.lo || b > self.hi {
            return None;
        }
        if self.need == 1 {
            Some(Utf8Tail::BOUNDARY)
        } else {
            tail(self.need - 1, 0x80, 0xBF)
        }
    }
}

/// A free-text part in progress (S6.6c): which part, how many bytes of its terminator the
/// text's tail matches, and the UTF-8 character being read.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct UntilItem {
    pub(crate) until: u32,
    pub(crate) k: u32,
    pub(crate) tail: Utf8Tail,
}

impl UntilItem {
    /// Whether the text stands between characters (no UTF-8 sequence is in progress).
    #[inline]
    pub(crate) fn between_characters(&self) -> bool {
        self.tail.need == 0
    }
}

pub(crate) enum UntilStep {
    /// The byte is text (or a piece of a terminator match in progress).
    More(UntilItem),
    /// The byte completed the terminator: the part is over.
    Done,
    /// The byte would make the text ill-formed UTF-8.
    No,
}

/// Feed one byte to a free-text part. The text and its terminator are one UTF-8 stream;
/// the part ends at the first byte that completes the terminator.
#[inline]
pub(crate) fn until_step(schema: &UntilSchema, item: UntilItem, b: u8) -> UntilStep {
    let Some(tail) = item.tail.step(b) else {
        return UntilStep::No;
    };
    let k = schema.next(item.k, b);
    if k as usize == schema.term.len() {
        return UntilStep::Done;
    }
    UntilStep::More(UntilItem {
        until: item.until,
        k,
        tail,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum Item {
    /// Expand this grammar node.
    Node(u32),
    /// Match exact bytes from `pos`.
    Lit {
        lit: u32,
        pos: u32,
    },
    Obj(ObjItem),
    Arr(ArrItem),
    Str(StrItem),
    Num(NumItem),
    /// Free text up to a terminator.
    Until(UntilItem),
    /// Repeat `inner`; `count` iterations have completed.
    Rep {
        inner: u32,
        count: u32,
        min: u32,
        max: u32,
    },
}

type Stack = Vec<Item>;

/// The incremental matcher: a compiled constraint plus the stacks its prefix leaves.
#[derive(Clone, Debug)]
pub struct Matcher {
    prog: Arc<Compiled>,
    stacks: Vec<Stack>,
}

impl PartialEq for Matcher {
    fn eq(&self, other: &Self) -> bool {
        // Two different compiled grammars can have identical instruction ids; the mask
        // cache must not serve one grammar's mask to the other.
        Arc::ptr_eq(&self.prog, &other.prog) && self.stacks == other.stacks
    }
}

impl Eq for Matcher {}

impl Hash for Matcher {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.prog) as usize).hash(state);
        self.stacks.hash(state);
    }
}

impl Matcher {
    /// A matcher at the start of the reply.
    pub fn new(constraint: &super::Constraint) -> Result<Matcher, super::SchemaError> {
        let prog = Arc::new(constraint.compile()?);
        let mut m = Matcher {
            prog,
            stacks: Vec::new(),
        };
        let mut out = BTreeSet::new();
        let mut budget = MAX_EXPANSIONS;
        if !m.close_into(vec![Item::Node(m.prog.root)], &mut out, &mut budget) || out.is_empty() {
            return Err(super::SchemaError::too_complex(
                "#",
                "it offers more alternatives at one point than the matcher follows at once \
                 (for example an enum or anyOf with tens of thousands of members); shorten or \
                 split it",
            ));
        }
        m.stacks = out.into_iter().collect();
        Ok(m)
    }

    /// The compiled grammar.
    pub fn compiled(&self) -> &Compiled {
        &self.prog
    }

    /// The bytes that may come next. Empty when the top-level value has closed, or when
    /// the matcher is poisoned by a schema too complex to serve.
    pub(crate) fn allowed_bytes(&self) -> ByteSet {
        let mut set = ByteSet::EMPTY;
        for stack in &self.stacks {
            set.union(self.stack_allowed(stack));
        }
        set
    }

    /// Whether `b` may come next.
    pub fn accepts(&self, b: u8) -> bool {
        self.allowed_bytes().contains(b)
    }

    /// Every byte that may come next, in ascending order (the recognised prefix's alphabet).
    pub fn accepted_bytes(&self) -> Vec<u8> {
        let set = self.allowed_bytes();
        (0..=255u8).filter(|&b| set.contains(b)).collect()
    }

    /// Feed one byte. `false` means no continuation accepts it.
    pub fn advance(&mut self, b: u8) -> bool {
        let mut out = BTreeSet::new();
        let mut budget = MAX_EXPANSIONS;
        for stack in &self.stacks {
            let mut raw = Vec::new();
            if !self.step_stack(stack, b, &mut raw, &mut budget) {
                self.stacks.clear();
                return false;
            }
            for s in raw {
                if !self.close_into(s, &mut out, &mut budget) {
                    self.stacks.clear();
                    return false;
                }
            }
        }
        if out.is_empty() {
            self.stacks.clear();
            return false;
        }
        self.stacks = out.into_iter().collect();
        true
    }

    /// Feed every byte of `bytes`; `Err(index)` is the byte that was rejected.
    pub fn advance_bytes(&mut self, bytes: &[u8]) -> Result<(), usize> {
        for (i, &b) in bytes.iter().enumerate() {
            if !self.advance(b) {
                return Err(i);
            }
        }
        Ok(())
    }

    /// Whether the reply may end here: every frame left can finish.
    pub fn can_end(&self) -> bool {
        self.stacks.iter().any(|s| self.stack_can_end(s))
    }

    /// Whether the top-level value has closed. Once it has, only end tokens may follow.
    pub fn is_closed(&self) -> bool {
        self.stacks.iter().any(Vec::is_empty)
    }

    /// Whether the state is "inside a plain string": one stack whose top is a string in
    /// its between-characters state. The mask uses this for its fast path.
    pub(crate) fn simple_string(&self) -> Option<(u32, u32, u32)> {
        if self.stacks.len() != 1 {
            return None;
        }
        match self.stacks[0].last()? {
            Item::Str(s) if s.state == StrState::Plain => {
                let schema = self.prog.strings[s.string as usize];
                Some((s.chars, schema.min, schema.max))
            }
            _ => None,
        }
    }

    /// The compiled grammar's identity: two matchers from different `Constraint`s never
    /// share a compiled arena, and a mask computed for one must not be served to the other.
    pub(crate) fn program_id(&self) -> usize {
        Arc::as_ptr(&self.prog) as usize
    }

    /// The structural key of the state, for a cache. Two matchers with the same key have
    /// the same allowed set.
    pub fn state_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    fn close_into(&self, stack: Stack, out: &mut BTreeSet<Stack>, budget: &mut usize) -> bool {
        let mut work = vec![stack];
        'work: while let Some(mut stack) = work.pop() {
            loop {
                if *budget == 0 {
                    return false;
                }
                *budget -= 1;
                let Some(top) = stack.last().copied() else {
                    out.insert(Vec::new());
                    break;
                };
                match top {
                    Item::Node(n) => {
                        stack.pop();
                        // An `Alt` replaces this path with one path per alternative;
                        // it is never an epsilon continuation of the parent.
                        if let Node::Alt(children) = &self.prog.nodes[n as usize] {
                            let children = children.clone();
                            for c in children {
                                let mut branch = stack.clone();
                                branch.push(Item::Node(c));
                                work.push(branch);
                            }
                            // This path is replaced by the branches.
                            continue 'work;
                        } else if !self.expand(n, &mut stack) {
                            return false;
                        }
                    }
                    Item::Lit { lit, pos } if pos as usize >= self.lit(lit).len() => {
                        stack.pop();
                    }
                    Item::Rep {
                        count, min, max, ..
                    } if count >= max && count >= min => {
                        stack.pop();
                    }
                    _ => {
                        out.insert(stack);
                        break;
                    }
                }
                if out.len() + work.len() > MAX_STACKS {
                    return false;
                }
            }
        }
        true
    }

    /// Expand one node onto `stack`; `Alt` branches into `work`.
    fn expand(&self, node: u32, stack: &mut Stack) -> bool {
        match &self.prog.nodes[node as usize] {
            Node::Lit(l) => {
                let bytes = &self.prog.lits[*l as usize];
                if !bytes.is_empty() {
                    stack.push(Item::Lit { lit: *l, pos: 0 });
                }
                true
            }
            Node::Seq(children) => {
                for &c in children.iter().rev() {
                    stack.push(Item::Node(c));
                }
                true
            }
            // `close_into` handles alternatives itself (they replace the path, not extend it).
            Node::Alt(_) => true,
            Node::Str(id) => {
                stack.push(Item::Str(StrItem {
                    string: *id,
                    state: StrState::Start,
                    chars: 0,
                }));
                true
            }
            Node::Until(id) => {
                stack.push(Item::Until(UntilItem {
                    until: *id,
                    k: 0,
                    tail: Utf8Tail::BOUNDARY,
                }));
                true
            }
            Node::Num(id) => {
                stack.push(Item::Num(NumItem {
                    number: *id,
                    state: NumState::Start,
                }));
                true
            }
            Node::Obj(id) => {
                stack.push(Item::Obj(ObjItem {
                    object: *id,
                    next: 0,
                    stage: ObjStage::Open,
                    ws: Ws::Start,
                }));
                true
            }
            Node::Arr(id) => {
                stack.push(Item::Arr(ArrItem {
                    array: *id,
                    index: 0,
                    stage: ArrStage::Open,
                    ws: Ws::Start,
                }));
                true
            }
            Node::Any => {
                stack.push(Item::Node(self.prog.any));
                true
            }
            Node::Repeat { inner, min, max } => {
                stack.push(Item::Rep {
                    inner: *inner,
                    count: 0,
                    min: *min,
                    max: *max,
                });
                true
            }
        }
    }

    #[inline]
    fn lit(&self, id: u32) -> &[u8] {
        &self.prog.lits[id as usize]
    }

    fn stack_allowed(&self, stack: &Stack) -> ByteSet {
        let mut set = ByteSet::EMPTY;
        for &item in stack.iter().rev() {
            match item {
                Item::Node(_) => break,
                Item::Lit { lit, pos } => {
                    if let Some(&b) = self.lit(lit).get(pos as usize) {
                        set.insert(b);
                    }
                    break;
                }
                Item::Obj(o) => {
                    set.union(self.obj_allowed(o));
                    break;
                }
                Item::Arr(a) => {
                    set.union(self.arr_allowed(a));
                    break;
                }
                Item::Str(s) => {
                    set.union(self.str_allowed(s));
                    break;
                }
                Item::Until(u) => {
                    if u.tail.need == 0 {
                        set.union(utf8_lead_bytes());
                    } else {
                        for b in u.tail.lo..=u.tail.hi {
                            set.insert(b);
                        }
                    }
                    break;
                }
                Item::Num(n) => {
                    set.union(self.num_allowed(n));
                    if !self.num_can_end(n) {
                        break;
                    }
                }
                Item::Rep {
                    inner,
                    count,
                    min,
                    max,
                } => {
                    if count < max {
                        set.union(self.prog.first_of(inner));
                    }
                    if count >= min {
                        continue;
                    }
                    break;
                }
            }
        }
        set
    }

    fn stack_can_end(&self, stack: &Stack) -> bool {
        for &item in stack.iter().rev() {
            match item {
                Item::Num(n) if self.num_can_end(n) => {}
                Item::Rep { count, min, .. } if count >= min => {}
                _ => return false,
            }
        }
        true
    }

    fn step_stack(&self, stack: &[Item], b: u8, out: &mut Vec<Stack>, budget: &mut usize) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let Some(&top) = stack.last() else {
            return true;
        };
        let below = || stack[..stack.len() - 1].to_vec();
        match top {
            Item::Node(_) => {
                // A closed stack never has a node on top; expand and retry.
                let mut closed = BTreeSet::new();
                if !self.close_into(stack.to_vec(), &mut closed, budget) {
                    return false;
                }
                for s in closed {
                    if !self.step_stack(&s, b, out, budget) {
                        return false;
                    }
                }
                true
            }
            Item::Lit { lit, pos } => {
                let bytes = self.lit(lit);
                if bytes.get(pos as usize) == Some(&b) {
                    let mut s = below();
                    if (pos as usize) + 1 < bytes.len() {
                        s.push(Item::Lit { lit, pos: pos + 1 });
                    }
                    out.push(s);
                }
                true
            }
            Item::Str(item) => {
                match self.str_step(item, b) {
                    StrStep::More(next) => {
                        let mut s = below();
                        s.push(Item::Str(next));
                        out.push(s);
                    }
                    StrStep::Done => out.push(below()),
                    StrStep::No => {}
                }
                true
            }
            Item::Until(item) => {
                match until_step(self.until_schema(item), item, b) {
                    UntilStep::More(next) => {
                        let mut s = below();
                        s.push(Item::Until(next));
                        out.push(s);
                    }
                    UntilStep::Done => out.push(below()),
                    UntilStep::No => {}
                }
                true
            }
            Item::Num(item) => {
                if let Some(next) = self.num_step(item, b) {
                    let mut s = below();
                    s.push(Item::Num(next));
                    out.push(s);
                }
                if self.num_can_end(item) {
                    return self.step_stack(&stack[..stack.len() - 1], b, out, budget);
                }
                true
            }
            Item::Rep {
                inner,
                count,
                min,
                max,
            } => {
                if count < max && self.prog.first_of(inner).contains(b) {
                    let mut s = below();
                    s.push(Item::Rep {
                        inner,
                        count: count + 1,
                        min,
                        max,
                    });
                    s.push(Item::Node(inner));
                    if !self.push_and_consume(s, b, out, budget) {
                        return false;
                    }
                }
                if count >= min {
                    return self.step_stack(&stack[..stack.len() - 1], b, out, budget);
                }
                true
            }
            Item::Obj(item) => self.obj_step(item, &below, b, out, budget),
            Item::Arr(item) => self.arr_step(item, &below, b, out, budget),
        }
    }

    fn push_and_consume(
        &self,
        stack: Stack,
        b: u8,
        out: &mut Vec<Stack>,
        budget: &mut usize,
    ) -> bool {
        let mut closed = BTreeSet::new();
        if !self.close_into(stack, &mut closed, budget) {
            return false;
        }
        for s in closed {
            if !self.step_stack(&s, b, out, budget) {
                return false;
            }
        }
        true
    }

    fn obj_allowed(&self, item: ObjItem) -> ByteSet {
        let object = &self.prog.objects[item.object as usize];
        let mut set = ByteSet::EMPTY;
        match item.stage {
            ObjStage::Open => set.insert(b'{'),
            ObjStage::KeyOrEnd => {
                set.union(item.ws.allowed());
                if (item.next as usize) < object.properties.len()
                    || object.additional != Additional::None
                {
                    set.insert(b'"');
                }
                if can_end_object(object, item.next) {
                    set.insert(b'}');
                }
            }
            ObjStage::AfterComma => {
                set.union(item.ws.allowed());
                set.insert(b'"');
            }
            ObjStage::AfterKey(_) => {
                set.union(item.ws.allowed());
                set.insert(b':');
            }
            ObjStage::BeforeValue(prop) => {
                set.union(item.ws.allowed());
                let node = object.properties[prop as usize].value;
                set.union(self.prog.first_of(node));
            }
            ObjStage::AfterValue => {
                set.union(item.ws.allowed());
                if can_take_more(object, item.next) {
                    set.insert(b',');
                }
                if can_end_object(object, item.next) {
                    set.insert(b'}');
                }
            }
            ObjStage::AfterAddKey => {
                set.union(item.ws.allowed());
                set.insert(b':');
            }
            ObjStage::BeforeAddValue => {
                set.union(item.ws.allowed());
                let node = additional_node(object);
                set.union(self.prog.first_of(node));
            }
        }
        set
    }

    fn obj_step(
        &self,
        item: ObjItem,
        below: &dyn Fn() -> Stack,
        b: u8,
        out: &mut Vec<Stack>,
        budget: &mut usize,
    ) -> bool {
        let object = &self.prog.objects[item.object as usize];
        if item.stage != ObjStage::Open {
            if let Some(ws) = item.ws.step(b) {
                let mut s = below();
                s.push(Item::Obj(ObjItem { ws, ..item }));
                out.push(s);
                return true;
            }
        }
        match item.stage {
            ObjStage::Open => {
                if b == b'{' {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::KeyOrEnd,
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                }
            }
            ObjStage::KeyOrEnd | ObjStage::AfterComma => {
                if b == b'"' {
                    // Every remaining declared property up to and including the next
                    // required one is a possible key.
                    // Additional keys follow all declared ones (the design's property
                    // order); an optional declared property must be emitted before any
                    // additional key, which keeps a declared name from taking the
                    // additional-properties schema.
                    let can_add = object.additional != Additional::None
                        && item.next as usize == object.properties.len();
                    if can_add {
                        let mut s = below();
                        s.push(Item::Obj(ObjItem {
                            stage: ObjStage::AfterAddKey,
                            next: object.properties.len() as u32,
                            ws: Ws::Start,
                            ..item
                        }));
                        s.push(Item::Str(StrItem {
                            string: self.prog.any_string,
                            state: StrState::Plain,
                            chars: 0,
                        }));
                        out.push(s);
                    }
                    if (item.next as usize) < object.properties.len() {
                        let last = next_required(object, item.next)
                            .unwrap_or(object.properties.len() as u32 - 1);
                        for prop in item.next..=last {
                            let mut s = below();
                            s.push(Item::Obj(ObjItem {
                                stage: ObjStage::AfterKey(prop),
                                next: prop + 1,
                                ws: Ws::Start,
                                ..item
                            }));
                            s.push(Item::Lit {
                                lit: object.properties[prop as usize].key,
                                pos: 1,
                            });
                            out.push(s);
                        }
                    }
                } else if b == b'}'
                    && item.stage == ObjStage::KeyOrEnd
                    && can_end_object(object, item.next)
                {
                    out.push(below());
                }
            }
            ObjStage::AfterKey(prop) => {
                if b == b':' {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::BeforeValue(prop),
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                }
            }
            ObjStage::BeforeValue(prop) => {
                let node = object.properties[prop as usize].value;
                if self.prog.first_of(node).contains(b) {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::AfterValue,
                        ws: Ws::Start,
                        ..item
                    }));
                    s.push(Item::Node(node));
                    return self.push_and_consume(s, b, out, budget);
                }
            }
            ObjStage::AfterValue => {
                if b == b',' && can_take_more(object, item.next) {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::AfterComma,
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                } else if b == b'}' && can_end_object(object, item.next) {
                    out.push(below());
                }
            }
            ObjStage::AfterAddKey => {
                if b == b':' {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::BeforeAddValue,
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                }
            }
            ObjStage::BeforeAddValue => {
                let node = additional_node(object);
                if self.prog.first_of(node).contains(b) {
                    let mut s = below();
                    s.push(Item::Obj(ObjItem {
                        stage: ObjStage::AfterValue,
                        ws: Ws::Start,
                        ..item
                    }));
                    s.push(Item::Node(node));
                    return self.push_and_consume(s, b, out, budget);
                }
            }
        }
        true
    }

    fn arr_allowed(&self, item: ArrItem) -> ByteSet {
        let array = &self.prog.arrays[item.array as usize];
        let mut set = ByteSet::EMPTY;
        match item.stage {
            ArrStage::Open => set.insert(b'['),
            ArrStage::FirstOrEnd => {
                set.union(item.ws.allowed());
                if array.min == 0 {
                    set.insert(b']');
                }
                if item.index < array.max {
                    set.union(self.prog.first_of(array_item(array, item.index)));
                }
            }
            ArrStage::AfterValue => {
                set.union(item.ws.allowed());
                if item.index < array.max {
                    set.insert(b',');
                }
                if item.index >= array.min {
                    set.insert(b']');
                }
            }
            ArrStage::AfterComma => {
                set.union(item.ws.allowed());
                if item.index < array.max {
                    set.union(self.prog.first_of(array_item(array, item.index)));
                }
            }
        }
        set
    }

    fn arr_step(
        &self,
        item: ArrItem,
        below: &dyn Fn() -> Stack,
        b: u8,
        out: &mut Vec<Stack>,
        budget: &mut usize,
    ) -> bool {
        let array = &self.prog.arrays[item.array as usize];
        if item.stage != ArrStage::Open {
            if let Some(ws) = item.ws.step(b) {
                let mut s = below();
                s.push(Item::Arr(ArrItem { ws, ..item }));
                out.push(s);
                return true;
            }
        }
        match item.stage {
            ArrStage::Open => {
                if b == b'[' {
                    let mut s = below();
                    s.push(Item::Arr(ArrItem {
                        stage: ArrStage::FirstOrEnd,
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                }
            }
            ArrStage::FirstOrEnd => {
                if b == b']' && array.min == 0 {
                    out.push(below());
                } else if item.index < array.max {
                    let node = array_item(array, item.index);
                    if self.prog.first_of(node).contains(b) {
                        let mut s = below();
                        s.push(Item::Arr(ArrItem {
                            stage: ArrStage::AfterValue,
                            index: item.index + 1,
                            ws: Ws::Start,
                            ..item
                        }));
                        s.push(Item::Node(node));
                        return self.push_and_consume(s, b, out, budget);
                    }
                }
            }
            ArrStage::AfterValue => {
                if b == b',' && item.index < array.max {
                    let mut s = below();
                    s.push(Item::Arr(ArrItem {
                        stage: ArrStage::AfterComma,
                        ws: Ws::Start,
                        ..item
                    }));
                    out.push(s);
                } else if b == b']' && item.index >= array.min {
                    out.push(below());
                }
            }
            ArrStage::AfterComma => {
                if item.index < array.max {
                    let node = array_item(array, item.index);
                    if self.prog.first_of(node).contains(b) {
                        let mut s = below();
                        s.push(Item::Arr(ArrItem {
                            stage: ArrStage::AfterValue,
                            index: item.index + 1,
                            ws: Ws::Start,
                            ..item
                        }));
                        s.push(Item::Node(node));
                        return self.push_and_consume(s, b, out, budget);
                    }
                }
            }
        }
        true
    }

    fn str_allowed(&self, item: StrItem) -> ByteSet {
        let schema = self.prog.strings[item.string as usize];
        let mut set = ByteSet::EMPTY;
        match item.state {
            StrState::Start => set.insert(b'"'),
            StrState::Plain => {
                if item.chars < schema.max {
                    // RFC 8259's unescaped characters: 0x20..0x7F minus `"` and `\`.
                    for b in 0x20u8..=0x7F {
                        if b != b'"' && b != b'\\' {
                            set.insert(b);
                        }
                    }
                    // The lead byte of a valid multi-byte sequence; the continuation
                    // ranges are checked in `str_step`.
                    for b in 0xC2u8..=0xF4 {
                        set.insert(b);
                    }
                    set.insert(b'\\');
                }
                if item.chars >= schema.min {
                    set.insert(b'"');
                }
            }
            StrState::Escape => {
                for &b in b"\"\\/bfnrtu" {
                    set.insert(b);
                }
            }
            StrState::EscapeU { .. } | StrState::LowU { .. } => {
                for b in b'0'..=b'9' {
                    set.insert(b);
                }
                for b in b'a'..=b'f' {
                    set.insert(b);
                }
                for b in b'A'..=b'F' {
                    set.insert(b);
                }
            }
            StrState::HighSlash => set.insert(b'\\'),
            StrState::HighU => set.insert(b'u'),
            StrState::Utf8 { lo, hi, .. } => {
                for b in lo..=hi {
                    set.insert(b);
                }
            }
        }
        set
    }

    fn str_step(&self, item: StrItem, b: u8) -> StrStep {
        str_step(self.prog.strings[item.string as usize], item, b)
    }

    /// The stack below a string's frame, when the state is one stack whose top is a
    /// string. The mask's string fast path walks a token trie with the string frame alone
    /// (a `Copy` state) instead of cloning the whole matcher at every trie node.
    pub(crate) fn string_split(&self) -> Option<(&[Item], StrItem)> {
        if self.stacks.len() != 1 {
            return None;
        }
        let stack = &self.stacks[0];
        match stack.last()? {
            Item::Str(item) => Some((&stack[..stack.len() - 1], *item)),
            _ => None,
        }
    }

    /// The string schema a string frame belongs to.
    pub(crate) fn string_schema(&self, item: StrItem) -> super::schema::StringSchema {
        self.prog.strings[item.string as usize]
    }

    /// The current string state's `(minLength, maxLength)`.
    pub(crate) fn string_bounds(&self) -> (u32, u32) {
        match self.string_split() {
            Some((_, item)) => {
                let s = self.string_schema(item);
                (s.min, s.max)
            }
            None => (0, u32::MAX),
        }
    }

    /// The same state with the string frame's character count folded into the categories
    /// that decide its transitions (`chars < min`, `min <= chars < max`, `chars >= max`).
    /// Masks that depend on the string only through those categories can be cached under
    /// this key; the exact count still filters the plain-token bitset.
    pub(crate) fn canonical_string(&self) -> Option<Matcher> {
        let (below, item) = self.string_split()?;
        let schema = self.string_schema(item);
        // Only an unbounded string's boundary transitions are independent of the exact
        // character count. A bounded `maxLength` changes which boundary tokens still fit,
        // so those states are cached by their exact count.
        if schema.max != u32::MAX {
            return None;
        }
        let repr = if item.chars < schema.min {
            0
        } else {
            schema.min
        };
        let mut stack = below.to_vec();
        stack.push(Item::Str(StrItem {
            chars: repr,
            ..item
        }));
        Some(Matcher {
            prog: self.prog.clone(),
            stacks: vec![stack],
        })
    }

    /// A matcher continuing from `below`, a stack that is already closed.
    pub(crate) fn continuing_below(&self, below: &[Item]) -> Matcher {
        Matcher {
            prog: self.prog.clone(),
            stacks: vec![below.to_vec()],
        }
    }

    /// The state "one stack whose top is a free-text part": the frames below it and the
    /// part's own item. The mask walks a token trie through free text with the item alone
    /// (a `Copy` state), as it does inside a string.
    pub(crate) fn until_split(&self) -> Option<(&[Item], UntilItem)> {
        if self.stacks.len() != 1 {
            return None;
        }
        let stack = &self.stacks[0];
        match stack.last()? {
            Item::Until(item) => Some((&stack[..stack.len() - 1], *item)),
            _ => None,
        }
    }

    /// The terminator table a free-text item belongs to.
    pub(crate) fn until_schema(&self, item: UntilItem) -> &UntilSchema {
        &self.prog.untils[item.until as usize]
    }

    /// The matcher that continues from the frames `below` a part that has just ended,
    /// with the next instructions expanded (the part after a free text is usually a node
    /// that has not been opened yet). `None` when the continuation is too large to follow.
    pub(crate) fn resume_below(&self, below: &[Item]) -> Option<Matcher> {
        let mut out = BTreeSet::new();
        let mut budget = MAX_EXPANSIONS;
        if !self.close_into(below.to_vec(), &mut out, &mut budget) {
            return None;
        }
        Some(Matcher {
            prog: self.prog.clone(),
            stacks: out.into_iter().collect(),
        })
    }

    /// Whether `bytes` can be read from here with **every byte consumed by a position the
    /// constraint spells**: a byte of a literal part, or a byte of a free-text part's
    /// terminator match. This is the rule for a marker token (S6.6c): a user-defined
    /// token such as `<tool_call>` may stand where the constraint writes those very
    /// bytes, and nowhere else — not as JSON content and not as free text.
    ///
    /// Inside a free-text part the first byte may begin the terminator's match (whatever
    /// the text's tail matched before); every later byte must extend that same match, so
    /// the whole token lies inside one occurrence of the terminator and what follows it.
    pub(crate) fn spelled(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() || self.stacks.is_empty() {
            return false;
        }
        let mut stacks = self.stacks.clone();
        for (i, &b) in bytes.iter().enumerate() {
            let mut out = BTreeSet::new();
            let mut budget = MAX_EXPANSIONS;
            for stack in &stacks {
                let mut raw = Vec::new();
                if !self.step_spelled(stack, b, i == 0, &mut raw, &mut budget) {
                    return false;
                }
                for s in raw {
                    if !self.close_into(s, &mut out, &mut budget) {
                        return false;
                    }
                }
            }
            if out.is_empty() {
                return false;
            }
            stacks = out.into_iter().collect();
        }
        true
    }

    /// [`step_stack`](Self::step_stack) restricted to spelled positions (see
    /// [`spelled`](Self::spelled)): a stack whose next byte would be JSON content or free
    /// text yields nothing.
    fn step_spelled(
        &self,
        stack: &[Item],
        b: u8,
        first: bool,
        out: &mut Vec<Stack>,
        budget: &mut usize,
    ) -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let Some(&top) = stack.last() else {
            return true;
        };
        let below = &stack[..stack.len() - 1];
        match top {
            Item::Node(_) => {
                let mut closed = BTreeSet::new();
                if !self.close_into(stack.to_vec(), &mut closed, budget) {
                    return false;
                }
                for s in closed {
                    if !self.step_spelled(&s, b, first, out, budget) {
                        return false;
                    }
                }
                true
            }
            Item::Lit { .. } => self.step_stack(stack, b, out, budget),
            Item::Until(item) => {
                let k = self.until_schema(item).next(item.k, b);
                let in_match = item.tail.need == 0 && k >= 1 && (first || k == item.k + 1);
                if in_match {
                    self.step_stack(stack, b, out, budget)
                } else {
                    true
                }
            }
            Item::Rep {
                inner,
                count,
                min,
                max,
            } => {
                if count < max && self.prog.first_of(inner).contains(b) {
                    let mut s = below.to_vec();
                    s.push(Item::Rep {
                        inner,
                        count: count + 1,
                        min,
                        max,
                    });
                    s.push(Item::Node(inner));
                    let mut closed = BTreeSet::new();
                    if !self.close_into(s, &mut closed, budget) {
                        return false;
                    }
                    for s in closed {
                        if !self.step_spelled(&s, b, first, out, budget) {
                            return false;
                        }
                    }
                }
                if count >= min {
                    return self.step_spelled(below, b, first, out, budget);
                }
                true
            }
            // JSON content is never a spelled position.
            Item::Obj(_) | Item::Arr(_) | Item::Str(_) | Item::Num(_) => true,
        }
    }

    fn num_allowed(&self, item: NumItem) -> ByteSet {
        let schema = self.prog.numbers[item.number as usize];
        let mut set = ByteSet::EMPTY;
        let digits = |set: &mut ByteSet| {
            set.insert(b'0');
            for b in b'1'..=b'9' {
                set.insert(b);
            }
        };
        if !schema.integer {
            // Fractions and exponents are JSON number syntax.
            match item.state {
                NumState::Zero => {
                    set.insert(b'.');
                    set.insert(b'e');
                    set.insert(b'E');
                }
                NumState::Int | NumState::Frac => {
                    digits(&mut set);
                    set.insert(b'.');
                    set.insert(b'e');
                    set.insert(b'E');
                }
                _ => {}
            }
        }
        match item.state {
            NumState::Start => {
                if schema.sign != Sign::NonNegative {
                    set.insert(b'-');
                }
                set.insert(b'0');
                if schema.sign != Sign::NonPositive {
                    for b in b'1'..=b'9' {
                        set.insert(b);
                    }
                }
            }
            NumState::Minus => digits(&mut set),
            NumState::Zero => {}
            NumState::Int => digits(&mut set),
            NumState::Dot => digits(&mut set),
            NumState::Frac => {
                digits(&mut set);
            }
            NumState::Exp => {
                set.insert(b'+');
                set.insert(b'-');
                digits(&mut set);
            }
            NumState::ExpSign => digits(&mut set),
            NumState::ExpDigits => digits(&mut set),
        }
        set
    }

    fn num_can_end(&self, item: NumItem) -> bool {
        matches!(
            item.state,
            NumState::Zero | NumState::Int | NumState::Frac | NumState::ExpDigits
        )
    }

    fn num_step(&self, item: NumItem, b: u8) -> Option<NumItem> {
        let schema = self.prog.numbers[item.number as usize];
        let next = match item.state {
            NumState::Start => match b {
                b'-' if schema.sign != Sign::NonNegative => NumState::Minus,
                b'0' => NumState::Zero,
                b'1'..=b'9' if schema.sign != Sign::NonPositive => NumState::Int,
                _ => return None,
            },
            NumState::Minus => match b {
                b'0' => NumState::Zero,
                b'1'..=b'9' => NumState::Int,
                _ => return None,
            },
            NumState::Zero => {
                if schema.integer {
                    return None;
                }
                match b {
                    b'.' => NumState::Dot,
                    b'e' | b'E' => NumState::Exp,
                    _ => return None,
                }
            }
            NumState::Int => match b {
                b'0'..=b'9' => NumState::Int,
                b'.' if !schema.integer => NumState::Dot,
                b'e' | b'E' if !schema.integer => NumState::Exp,
                _ => return None,
            },
            NumState::Dot => match b {
                b'0'..=b'9' => NumState::Frac,
                _ => return None,
            },
            NumState::Frac => match b {
                b'0'..=b'9' => NumState::Frac,
                b'e' | b'E' if !schema.integer => NumState::Exp,
                _ => return None,
            },
            NumState::Exp => match b {
                b'+' | b'-' => NumState::ExpSign,
                b'0'..=b'9' => NumState::ExpDigits,
                _ => return None,
            },
            NumState::ExpSign => match b {
                b'0'..=b'9' => NumState::ExpDigits,
                _ => return None,
            },
            NumState::ExpDigits => match b {
                b'0'..=b'9' => NumState::ExpDigits,
                _ => return None,
            },
        };
        Some(NumItem {
            state: next,
            ..item
        })
    }
}

/// The string frame's transition on one byte, shared by the matcher and the mask's
/// allocation-free string walk.
pub(crate) fn str_step(schema: super::schema::StringSchema, item: StrItem, b: u8) -> StrStep {
    match item.state {
        StrState::Start => {
            if b == b'"' {
                StrStep::More(StrItem {
                    state: StrState::Plain,
                    ..item
                })
            } else {
                StrStep::No
            }
        }
        StrState::Plain => {
            if b == b'"' {
                if item.chars >= schema.min {
                    StrStep::Done
                } else {
                    StrStep::No
                }
            } else if b == b'\\' {
                if item.chars < schema.max {
                    StrStep::More(StrItem {
                        state: StrState::Escape,
                        ..item
                    })
                } else {
                    StrStep::No
                }
            } else if (0x20..=0x7F).contains(&b) {
                if item.chars < schema.max {
                    StrStep::More(StrItem {
                        state: StrState::Plain,
                        chars: item.chars + 1,
                        ..item
                    })
                } else {
                    StrStep::No
                }
            } else if let Some((need, lo, hi)) = utf8_lead(b) {
                if item.chars < schema.max {
                    StrStep::More(StrItem {
                        state: StrState::Utf8 { need, lo, hi },
                        ..item
                    })
                } else {
                    StrStep::No
                }
            } else {
                StrStep::No
            }
        }
        StrState::Escape => match b {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                if item.chars < schema.max {
                    StrStep::More(StrItem {
                        state: StrState::Plain,
                        chars: item.chars + 1,
                        ..item
                    })
                } else {
                    StrStep::No
                }
            }
            b'u' => StrStep::More(StrItem {
                state: StrState::EscapeU {
                    digits: 0,
                    value: 0,
                },
                ..item
            }),
            _ => StrStep::No,
        },
        StrState::EscapeU { digits, value } => {
            let Some(d) = hex(b) else { return StrStep::No };
            let value = (value << 4) | d as u16;
            if digits < 3 {
                // A first `\u` code unit may not be a lone low surrogate. Prune a
                // prefix whose every completion would be one: otherwise the
                // automaton would reach a state with no allowed byte at all.
                let remaining = 3 - digits;
                let base = (value as u32) << (4 * remaining);
                let span = (1u32 << (4 * remaining)) - 1;
                if base >= 0xDC00 && base + span <= 0xDFFF {
                    return StrStep::No;
                }
                StrStep::More(StrItem {
                    state: StrState::EscapeU {
                        digits: digits + 1,
                        value,
                    },
                    ..item
                })
            } else if (0xD800..0xDC00).contains(&value) {
                StrStep::More(StrItem {
                    state: StrState::HighSlash,
                    ..item
                })
            } else if (0xDC00..0xE000).contains(&value) {
                // A lone low surrogate.
                StrStep::No
            } else if item.chars < schema.max {
                StrStep::More(StrItem {
                    state: StrState::Plain,
                    chars: item.chars + 1,
                    ..item
                })
            } else {
                StrStep::No
            }
        }
        StrState::HighSlash => {
            if b == b'\\' {
                StrStep::More(StrItem {
                    state: StrState::HighU,
                    ..item
                })
            } else {
                StrStep::No
            }
        }
        StrState::HighU => {
            if b == b'u' {
                StrStep::More(StrItem {
                    state: StrState::LowU {
                        digits: 0,
                        value: 0,
                    },
                    ..item
                })
            } else {
                StrStep::No
            }
        }
        StrState::LowU { digits, value } => {
            let Some(d) = hex(b) else { return StrStep::No };
            let value = (value << 4) | d as u16;
            if digits < 3 {
                let remaining = 3 - digits;
                let base = (value as u32) << (4 * remaining);
                let span = (1u32 << (4 * remaining)) - 1;
                // Every completion must be a low surrogate; prune the prefix that
                // cannot reach one.
                if base + span < 0xDC00 || base > 0xDFFF {
                    return StrStep::No;
                }
                StrStep::More(StrItem {
                    state: StrState::LowU {
                        digits: digits + 1,
                        value,
                    },
                    ..item
                })
            } else if (0xDC00..0xE000).contains(&value) && item.chars < schema.max {
                StrStep::More(StrItem {
                    state: StrState::Plain,
                    chars: item.chars + 1,
                    ..item
                })
            } else {
                StrStep::No
            }
        }
        StrState::Utf8 { need, lo, hi } => {
            if b < lo || b > hi {
                return StrStep::No;
            }
            let need = need - 1;
            if need == 0 {
                if item.chars < schema.max {
                    StrStep::More(StrItem {
                        state: StrState::Plain,
                        chars: item.chars + 1,
                        ..item
                    })
                } else {
                    StrStep::No
                }
            } else {
                StrStep::More(StrItem {
                    state: StrState::Utf8 {
                        need,
                        lo: 0x80,
                        hi: 0xBF,
                    },
                    ..item
                })
            }
        }
    }
}

pub(crate) enum StrStep {
    More(StrItem),
    Done,
    No,
}

fn can_end_object(object: &super::schema::ObjectSchema, next: u32) -> bool {
    object
        .properties
        .iter()
        .skip(next as usize)
        .all(|p| !p.required)
}

fn can_take_more(object: &super::schema::ObjectSchema, next: u32) -> bool {
    (next as usize) < object.properties.len() || object.additional != Additional::None
}

fn next_required(object: &super::schema::ObjectSchema, from: u32) -> Option<u32> {
    object
        .properties
        .iter()
        .enumerate()
        .skip(from as usize)
        .find(|(_, p)| p.required)
        .map(|(i, _)| i as u32)
}

fn additional_node(object: &super::schema::ObjectSchema) -> u32 {
    match object.additional {
        Additional::Node(n) => n,
        Additional::Any => 0,
        Additional::None => 0,
    }
}

fn array_item(array: &super::schema::ArraySchema, index: u32) -> u32 {
    array
        .prefix
        .get(index as usize)
        .copied()
        .unwrap_or(array.items)
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// The lead byte's continuation count and the first continuation's range.
fn utf8_lead(b: u8) -> Option<(u8, u8, u8)> {
    match b {
        0xC2..=0xDF => Some((1, 0x80, 0xBF)),
        0xE0 => Some((2, 0xA0, 0xBF)),
        0xE1..=0xEC => Some((2, 0x80, 0xBF)),
        0xED => Some((2, 0x80, 0x9F)),
        0xEE..=0xEF => Some((2, 0x80, 0xBF)),
        0xF0 => Some((3, 0x90, 0xBF)),
        0xF1..=0xF3 => Some((3, 0x80, 0xBF)),
        0xF4 => Some((3, 0x80, 0x8F)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constrain::{Constraint, Part, Schema};
    use crate::json::Value as J;

    fn schema(text: &str) -> Schema {
        Schema::new(J::parse(text).unwrap())
    }

    fn matcher(c: &Constraint) -> Matcher {
        Matcher::new(c).unwrap()
    }

    fn accepted(m: &mut Matcher, text: &[u8]) -> bool {
        m.advance_bytes(text).is_ok()
    }

    // ---- free text (S6.6c) ----

    fn until(term: &str, after: &str) -> Constraint {
        Constraint::sequence(vec![Part::Until(term.into()), Part::Literal(after.into())])
    }

    /// Whether `text` is a whole reply of `c`.
    fn whole(c: &Constraint, text: &[u8]) -> bool {
        let mut m = matcher(c);
        accepted(&mut m, text) && m.is_closed()
    }

    #[test]
    fn free_text_ends_at_the_first_occurrence_of_its_terminator() {
        let c = until("</think>", "!");
        assert!(whole(&c, b"some thoughts</think>!"));
        // The text may be empty, and may hold pieces of the terminator.
        assert!(whole(&c, b"</think>!"));
        assert!(whole(&c, b"</thi</think>!"));
        assert!(whole(&c, b"a < b, a </ b, </think </think>!"));
        // The part is over at the first occurrence: what follows is the literal, not text.
        assert!(!whole(&c, b"a</think>b</think>!"));
        let mut m = matcher(&c);
        assert!(accepted(&mut m, b"a</think>"));
        assert_eq!(m.accepted_bytes(), vec![b'!']);
        // A terminator whose prefix repeats: the search must fall back, not restart.
        let c = until("aab", "!");
        assert!(whole(&c, b"aaab!"));
        assert!(whole(&c, b"aaaaab!"));
        assert!(whole(&c, b"abaab!"));
        assert!(!whole(&c, b"aabb!"));
        let c = until("abab", ".");
        assert!(whole(&c, b"abab."));
        assert!(
            !whole(&c, b"ababab."),
            "the first occurrence ends at the fourth byte, and the text does not go on"
        );
        assert!(whole(&c, b"aabab."));
    }

    #[test]
    fn a_reply_cannot_end_inside_free_text() {
        let c = Constraint::sequence(vec![Part::Until("END".into())]);
        let mut m = matcher(&c);
        for chunk in [&b"text"[..], b" E", b"N", b"x EN"] {
            assert!(accepted(&mut m, chunk));
            assert!(!m.can_end() && !m.is_closed(), "after {chunk:?}");
        }
        assert!(accepted(&mut m, b"D"));
        assert!(m.can_end() && m.is_closed());
        // A reply that never writes the terminator never becomes acceptable.
        let mut m = matcher(&c);
        for _ in 0..500 {
            assert!(accepted(&mut m, b"EN ND E"));
            assert!(!m.can_end());
        }
    }

    #[test]
    fn free_text_is_well_formed_utf8() {
        let c = until("</t>", "");
        // Multi-byte characters, fed whole and one byte at a time.
        let text = "é ✓ 🦅</t>".as_bytes();
        assert!(whole(&c, text));
        let mut m = matcher(&c);
        for &b in text {
            assert!(m.advance(b));
        }
        assert!(m.is_closed());
        // Ill-formed sequences are refused at the byte that makes them so.
        let refused_at = |bytes: &[u8]| matcher(&c).advance_bytes(bytes).err();
        assert_eq!(refused_at(b"\xFF"), Some(0));
        assert_eq!(
            refused_at(b"a\x80"),
            Some(1),
            "a continuation byte with no lead"
        );
        assert_eq!(refused_at(b"\xC0\xAF"), Some(0), "an overlong form");
        assert_eq!(
            refused_at(b"\xE0\x80\x80"),
            Some(1),
            "an overlong three-byte form"
        );
        assert_eq!(refused_at(b"\xED\xA0\x80"), Some(1), "a surrogate");
        assert_eq!(refused_at(b"\xF4\x90\x80\x80"), Some(1), "above U+10FFFF");
        assert_eq!(
            refused_at(b"\xC3</t>"),
            Some(1),
            "a character cut short by the terminator"
        );
        // Inside a character only its continuation bytes are offered.
        let mut m = matcher(&c);
        assert!(accepted(&mut m, b"\xE2\x9C"));
        assert_eq!(m.accepted_bytes(), (0x80..=0xBFu8).collect::<Vec<_>>());
    }

    #[test]
    fn free_text_composes_with_the_other_parts() {
        let call = Constraint::sequence(vec![
            Part::Until("</think>".into()),
            Part::Literal("\n\n<tool_call>\n".into()),
            Part::Json(schema(
                r#"{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false}"#,
            )),
            Part::Literal("\n</tool_call>".into()),
        ]);
        assert!(whole(
            &call,
            b"hm, {\"n\": oops}</think>\n\n<tool_call>\n{\"n\": 3}\n</tool_call>"
        ));
        assert!(!whole(
            &call,
            b"hm</think>\n\n<tool_call>\n{\"n\": \"3\"}\n</tool_call>"
        ));
        // Optional (a repetition of none or one) and beside an alternative.
        let c = Constraint::concat(vec![
            Constraint::repeat(
                Constraint::sequence(vec![Part::Until("|".into())]),
                0,
                Some(1),
            ),
            Constraint::any_of(vec![
                Constraint::sequence(vec![Part::Literal("yes".into())]),
                Constraint::sequence(vec![Part::Until(";".into()), Part::Literal("no".into())]),
            ]),
        ]);
        for ok in [
            &b"yes"[..],
            b"thinking|yes",
            b"|yes",
            b"maybe;no",
            b"a|b;no",
            b"yes;no",
        ] {
            assert!(whole(&c, ok), "{:?}", std::str::from_utf8(ok));
        }
        for bad in [&b"ye"[..], b"a|b|yes", b"maybe;yes", b"a|b;n"] {
            assert!(!whole(&c, bad), "{:?}", std::str::from_utf8(bad));
        }
        // An empty terminator is refused when the constraint is compiled.
        let e = Constraint::sequence(vec![Part::Until(String::new())])
            .compile()
            .expect_err("refused");
        assert_eq!(e.keyword, "until");
    }

    #[test]
    fn spelled_bytes_are_literal_bytes_and_terminator_bytes() {
        let c = Constraint::sequence(vec![
            Part::Until("</t>".into()),
            Part::Literal("\n<c>".into()),
            Part::Json(schema(r#"{"type":"string"}"#)),
        ]);
        let at = |prefix: &[u8]| {
            let mut m = matcher(&c);
            assert!(accepted(&mut m, prefix));
            m
        };
        // In free text: the terminator, a piece of it from its start, the terminator and
        // the literal after it in one piece. Never text, never text and then terminator.
        let m = at(b"abc");
        assert!(m.spelled(b"</t>") && m.spelled(b"</") && m.spelled(b"</t>\n<c>"));
        assert!(!m.spelled(b"x") && !m.spelled(b"x</t>") && !m.spelled(b"</t>z"));
        assert!(!m.spelled(b"/t>"), "not from the terminator's start");
        // With a match in progress a piece may extend it, or the terminator may start anew.
        let m = at(b"abc<");
        assert!(m.spelled(b"/t>") && m.spelled(b"</t>"));
        assert!(!m.spelled(b"t>"));
        // In the literal: its bytes, from where it stands; not past it into the JSON.
        let m = at(b"abc</t>");
        assert!(m.spelled(b"\n<c>") && m.spelled(b"\n<"));
        assert!(!m.spelled(b"<c>") && !m.spelled(b"\n<c>\""));
        // In the JSON string nothing is spelled.
        let m = at(b"abc</t>\n<c>\"so ");
        assert!(!m.spelled(b"</t>") && !m.spelled(b"<c>") && !m.spelled(b"\""));
        // A literal at the head of a repeated part is spelled at the repetition's edge.
        let c = Constraint::repeat(
            Constraint::sequence(vec![
                Part::Literal("<x>".into()),
                Part::Json(schema(r#"{"const":1}"#)),
            ]),
            1,
            Some(3),
        );
        let mut m = matcher(&c);
        assert!(m.spelled(b"<x>"));
        assert!(accepted(&mut m, b"<x>1"));
        assert!(m.spelled(b"<x>") && !m.spelled(b"1"));
    }

    #[test]
    fn a_literal_is_matched_byte_by_byte() {
        let c = Constraint::sequence(vec![Part::Literal("<<A>>".into())]);
        let mut m = matcher(&c);
        assert!(accepted(&mut m, b"<<A>>"));
        assert!(m.is_closed());
        assert!(!accepted(&mut m, b"x"));
    }

    #[test]
    fn a_json_object_is_accepted_and_a_comma_must_be_followed_by_a_key() {
        let c = Constraint::from_schema(schema(r#"{"type":"object"}"#));
        let mut m = matcher(&c);
        assert!(accepted(&mut m, br#"{"a":1}"#));
        assert!(m.can_end());
        assert!(!accepted(&mut matcher(&c), br#"{"a":1,}"#));
    }

    #[test]
    fn whitespace_is_bounded() {
        let c = Constraint::from_schema(schema(r#"{"type":"array"}"#));
        assert!(accepted(&mut matcher(&c), b"[ 1 , 2 ]"));
        assert!(accepted(&mut matcher(&c), b"[\n \t1\n\n]"));
        assert!(!accepted(&mut matcher(&c), b"[ 1  ]"));
        assert!(!accepted(&mut matcher(&c), b"[ \t1]"));
        assert!(!accepted(
            &mut matcher(&c),
            format!("[\n{}1]", " ".repeat(21)).as_bytes()
        ));
    }

    #[test]
    fn strings_track_utf8_and_escapes_across_bytes() {
        let c = Constraint::from_schema(schema(r#"{"type":"string"}"#));
        let mut m = matcher(&c);
        assert!(accepted(&mut m, b"\"\xC3"));
        assert!(accepted(&mut m, b"\xA9\""));
        assert!(!accepted(&mut m, b"\xED\xA0\x80\""));
        assert!(accepted(&mut matcher(&c), "\"\\ud83d\\ude00\"".as_bytes()));
        assert!(!accepted(&mut matcher(&c), "\"\\ud83d\"".as_bytes()));
    }

    #[test]
    fn a_tree_ref_recurses() {
        let c = Constraint::from_schema(schema(
            r##"{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#"}}},"required":["v"],"additionalProperties":false}"##,
        ));
        let mut m = matcher(&c);
        assert!(accepted(&mut m, br#"{"v":1,"kids":[{"v":2}]}"#));
        assert!(m.is_closed());
    }

    #[test]
    fn an_anyof_overlap_keeps_both_alternatives() {
        let c = Constraint::from_schema(schema(r#"{"anyOf":[{"const":1},{"const":10}]}"#));
        let mut m = matcher(&c);
        assert!(m.accepts(b'1'));
        assert!(m.advance(b'1'));
        assert!(m.accepts(b'0'));
        assert!(m.advance(b'0'));
        assert!(m.is_closed());
    }
}
