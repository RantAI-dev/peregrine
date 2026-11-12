//! JSON Schema (the supported subset) → the byte-level grammar the matcher walks.
//!
//! The API is [`Constraint`]: a tree of literals, JSON values and combinators. Building one
//! is cheap (it only owns the schema values); [`Constraint::compile`] lowers it to a single
//! instruction arena ([`Compiled`]) that the [matcher](super::Matcher) runs.
//!
//! # What is supported, and what is refused
//!
//! The supported keyword set is listed in `docs/design/s6.6-constrained-decoding.md` §2.
//! Anything else that could restrict the value is refused with [`SchemaError`], naming the
//! keyword and the JSON pointer, never silently ignored: a client that relies on `pattern`
//! or `format` would otherwise receive a reply that violates its schema.
//!
//! # Property order
//!
//! An object's declared properties are emitted in the order of the schema's `properties`
//! object; a required property must appear; an optional one may be skipped only while no
//! later required property forces it out. Additional properties, where allowed, follow the
//! declared ones. OpenAI's structured outputs do the same, and it keeps the automaton small.

use crate::json::Value;
use std::collections::HashMap;
use std::fmt;

/// The most values a bounded integer range may hold; above it the keyword is refused
/// rather than expanding a thousand-way choice into a thousand literals.
pub const MAX_FINITE_INT_VALUES: i128 = 1000;

/// A schema error: the keyword that is not honoured, where it appears, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    /// The keyword that was refused (`"pattern"`, `"$ref"`, …), or a synthetic name for a
    /// combinator misuse (`"repeat"`).
    pub keyword: String,
    /// The JSON pointer of the schema node, `#` for the root.
    pub pointer: String,
    /// A sentence for the client.
    pub detail: String,
}

impl SchemaError {
    pub(crate) fn new(
        keyword: impl Into<String>,
        pointer: impl Into<String>,
        detail: impl Into<String>,
    ) -> SchemaError {
        SchemaError {
            keyword: keyword.into(),
            pointer: pointer.into(),
            detail: detail.into(),
        }
    }

    /// A schema whose keywords are all supported but which is too large to serve: no single
    /// keyword is at fault, so the message names the place and the reason only.
    pub(crate) fn too_complex(
        pointer: impl Into<String>,
        detail: impl Into<String>,
    ) -> SchemaError {
        SchemaError {
            keyword: String::new(),
            pointer: pointer.into(),
            detail: detail.into(),
        }
    }

    /// The message an OpenAI-shaped 400 carries.
    pub fn message(&self) -> String {
        if self.keyword.is_empty() {
            return format!(
                "the JSON Schema at {} cannot be served: {}",
                self.pointer, self.detail
            );
        }
        format!(
            "unsupported JSON Schema keyword {:?} at {}: {}",
            self.keyword, self.pointer, self.detail
        )
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for SchemaError {}

/// A parsed JSON Schema, owned.
///
/// The schema document is not interpreted until the [`Constraint`] that holds it is
/// compiled, so a combinator can be built without re-parsing.
#[derive(Debug, Clone)]
pub struct Schema {
    pub(crate) value: Value,
    pub(crate) strict: bool,
}

impl Schema {
    /// A schema as JSON Schema defines it (`additionalProperties` defaults to true).
    pub fn new(value: Value) -> Schema {
        Schema {
            value,
            strict: false,
        }
    }

    /// OpenAI's `"strict": true`: `additionalProperties` is treated as false and every
    /// declared property is required.
    pub fn with_strict(value: Value, strict: bool) -> Schema {
        Schema { value, strict }
    }

    /// The schema document.
    pub fn value(&self) -> &Value {
        &self.value
    }
}

/// One part of a [`Constraint::sequence`]: exact bytes, or a JSON value shaped by a schema.
#[derive(Debug, Clone)]
pub enum Part {
    /// Matched byte-exactly (so a token may end in the middle of it).
    Literal(String),
    /// One JSON value of this schema.
    Json(Schema),
}

#[derive(Debug, Clone)]
pub(crate) enum Ast {
    Lit(Vec<u8>),
    Json(Schema),
    Seq(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat(Box<Ast>, u32, u32),
    JsonObject,
}

/// A constraint on the assistant's reply text.
///
/// Build one with [`sequence`](Constraint::sequence), [`any_of`](Constraint::any_of) or
/// [`repeat`](Constraint::repeat), or from a single schema with
/// [`from_schema`](Constraint::from_schema). Attach it to a generation request with
/// `SlotCfg::constraint`; the server's `response_format` uses the same path.
#[derive(Debug, Clone)]
pub struct Constraint {
    pub(crate) ast: Ast,
}

impl Constraint {
    /// Literal parts and JSON values, in order, matched byte for byte.
    pub fn sequence(parts: Vec<Part>) -> Constraint {
        let parts = parts
            .into_iter()
            .map(|p| match p {
                Part::Literal(s) => Ast::Lit(s.into_bytes()),
                Part::Json(s) => Ast::Json(s),
            })
            .collect();
        Constraint {
            ast: Ast::Seq(parts),
        }
    }

    /// One of several constraints. Overlapping prefixes are handled: the matcher keeps
    /// every alternative that is still possible.
    pub fn any_of(alts: Vec<Constraint>) -> Constraint {
        Constraint {
            ast: Ast::Alt(alts.into_iter().map(|c| c.ast).collect()),
        }
    }

    /// The same constraint repeated `min..=max` times (`max: None` = unbounded).
    ///
    /// The inner constraint must consume at least one byte, or the automaton would loop
    /// without advancing; compiling a nullable inner constraint is refused.
    pub fn repeat(inner: Constraint, min: usize, max: Option<usize>) -> Constraint {
        Constraint {
            ast: Ast::Repeat(
                Box::new(inner.ast),
                min.min(u32::MAX as usize) as u32,
                max.map_or(u32::MAX, |m| m.min(u32::MAX as usize) as u32),
            ),
        }
    }

    /// Any JSON object: the language of `{"type": "object"}`.
    pub fn json_object() -> Constraint {
        Constraint {
            ast: Ast::JsonObject,
        }
    }

    /// Exactly one JSON value of `schema`.
    pub fn from_schema(schema: Schema) -> Constraint {
        Constraint {
            ast: Ast::Json(schema),
        }
    }

    /// Lower the constraint to the matcher's instruction arena.
    pub fn compile(&self) -> Result<Compiled, SchemaError> {
        let mut c = Compiler::new();
        let root = c.ast(&self.ast)?;
        c.out.root = root;
        c.out.analyse()?;
        Ok(c.out)
    }
}

/// A 256-bit set of bytes, the matcher's alphabet.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ByteSet(pub(crate) [u64; 4]);

impl ByteSet {
    pub(crate) const EMPTY: ByteSet = ByteSet([0; 4]);

    #[inline]
    pub(crate) fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    #[inline]
    pub(crate) fn contains(self, b: u8) -> bool {
        self.0[(b >> 6) as usize] & (1u64 << (b & 63)) != 0
    }

    pub(crate) fn union(&mut self, other: ByteSet) {
        for i in 0..4 {
            self.0[i] |= other.0[i];
        }
    }
}

impl fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ByteSet[")?;
        let mut first = true;
        for b in 0..=255u8 {
            if self.contains(b) {
                if !first {
                    f.write_str(",")?;
                }
                first = false;
                write!(f, "{b}")?;
            }
        }
        f.write_str("]")
    }
}

/// How an object's extra keys are treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Additional {
    None,
    Any,
    Node(u32),
}

#[derive(Debug, Clone)]
pub(crate) struct Property {
    pub(crate) key: u32,
    pub(crate) value: u32,
    pub(crate) required: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ObjectSchema {
    pub(crate) properties: Vec<Property>,
    pub(crate) additional: Additional,
}

#[derive(Debug, Clone)]
pub(crate) struct ArraySchema {
    pub(crate) prefix: Vec<u32>,
    /// The item schema after the prefix (`Any` when the schema has no `items`).
    pub(crate) items: u32,
    pub(crate) min: u32,
    pub(crate) max: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StringSchema {
    pub(crate) min: u32,
    pub(crate) max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sign {
    Any,
    NonNegative,
    NonPositive,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NumberSchema {
    pub(crate) integer: bool,
    pub(crate) sign: Sign,
}

/// One instruction of the compiled grammar.
#[derive(Debug, Clone)]
pub(crate) enum Node {
    /// Exact bytes (`lits` entry).
    Lit(u32),
    /// Concatenation.
    Seq(Vec<u32>),
    /// One of several alternatives.
    Alt(Vec<u32>),
    /// A JSON string with length bounds.
    Str(u32),
    /// A JSON number/integer.
    Num(u32),
    /// A JSON object.
    Obj(u32),
    /// A JSON array.
    Arr(u32),
    /// Any JSON value.
    Any,
    /// `inner` repeated `min..=max` times (`max == u32::MAX` = unbounded).
    Repeat { inner: u32, min: u32, max: u32 },
}

/// The compiled constraint: one arena of instructions and the tables they name.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub(crate) nodes: Vec<Node>,
    pub(crate) lits: Vec<Vec<u8>>,
    pub(crate) objects: Vec<ObjectSchema>,
    pub(crate) arrays: Vec<ArraySchema>,
    pub(crate) strings: Vec<StringSchema>,
    pub(crate) numbers: Vec<NumberSchema>,
    pub(crate) any: u32,
    pub(crate) any_string: u32,
    pub(crate) root: u32,
    /// First byte of a node's language, and the shortest byte length it can match.
    pub(crate) first: Vec<ByteSet>,
    pub(crate) min_len: Vec<u32>,
}

impl Compiled {
    fn push(&mut self, node: Node) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        id
    }

    /// Intern a literal's bytes and return its table index.
    fn lit_id(&mut self, bytes: Vec<u8>) -> u32 {
        let id = self.lits.len() as u32;
        self.lits.push(bytes);
        id
    }

    /// A node that matches an exact byte string.
    fn lit(&mut self, bytes: Vec<u8>) -> u32 {
        let id = self.lit_id(bytes);
        self.push(Node::Lit(id))
    }

    /// Compute [`first`](Self::first) and [`min_len`](Self::min_len) to a fixed point.
    fn analyse(&mut self) -> Result<(), SchemaError> {
        let n = self.nodes.len();
        let mut min_len = vec![u32::MAX; n];
        loop {
            let mut changed = false;
            for i in 0..n {
                let v = match &self.nodes[i] {
                    Node::Lit(l) => self.lits[*l as usize].len().min(u32::MAX as usize) as u32,
                    Node::Str(_) | Node::Num(_) | Node::Obj(_) | Node::Arr(_) | Node::Any => 1,
                    Node::Seq(children) => {
                        let mut sum = 0u32;
                        for &c in children {
                            match min_len[c as usize] {
                                u32::MAX => {
                                    sum = u32::MAX;
                                    break;
                                }
                                x => sum = sum.saturating_add(x),
                            }
                        }
                        sum
                    }
                    Node::Alt(children) => children
                        .iter()
                        .map(|&c| min_len[c as usize])
                        .min()
                        .unwrap_or(u32::MAX),
                    Node::Repeat { inner, min, .. } => {
                        if *min == 0 {
                            0
                        } else {
                            min_len[*inner as usize]
                        }
                    }
                };
                if v < min_len[i] {
                    min_len[i] = v;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if min_len[self.root as usize] == u32::MAX {
            return Err(SchemaError::new(
                "type",
                "#",
                "the schema matches no JSON value (every alternative is unsatisfiable)",
            ));
        }
        // A repeat of something that can match nothing would loop without consuming a byte.
        for (i, node) in self.nodes.iter().enumerate() {
            if let Node::Repeat { inner, .. } = node {
                if min_len[*inner as usize] == 0 {
                    return Err(SchemaError::new(
                        "repeat",
                        "#",
                        format!(
                            "the constraint repeated at node {i} can match empty, so the automaton would not advance"
                        ),
                    ));
                }
            }
        }
        let mut first = vec![ByteSet::EMPTY; n];
        loop {
            let mut changed = false;
            for i in 0..n {
                let mut set = ByteSet::EMPTY;
                match &self.nodes[i] {
                    Node::Lit(l) => {
                        if let Some(&b) = self.lits[*l as usize].first() {
                            set.insert(b);
                        }
                    }
                    Node::Str(_) => set.insert(b'"'),
                    Node::Num(num) => {
                        let s = self.numbers[*num as usize];
                        if s.sign != Sign::NonNegative {
                            set.insert(b'-');
                        }
                        set.insert(b'0');
                        if s.sign != Sign::NonPositive {
                            for b in b'1'..=b'9' {
                                set.insert(b);
                            }
                        }
                    }
                    Node::Obj(_) => set.insert(b'{'),
                    Node::Arr(_) => set.insert(b'['),
                    Node::Any => set = first[self.any as usize],
                    Node::Seq(children) => {
                        for &c in children {
                            set.union(first[c as usize]);
                            if min_len[c as usize] > 0 {
                                break;
                            }
                        }
                    }
                    Node::Alt(children) => {
                        for &c in children {
                            if min_len[c as usize] != u32::MAX {
                                set.union(first[c as usize]);
                            }
                        }
                    }
                    Node::Repeat { inner, max, .. } => {
                        if *max > 0 {
                            set = first[*inner as usize];
                        }
                    }
                }
                let before = first[i];
                let mut merged = before;
                merged.union(set);
                if merged != before {
                    first[i] = merged;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        self.first = first;
        self.min_len = min_len;
        Ok(())
    }

    #[inline]
    pub(crate) fn first_of(&self, node: u32) -> ByteSet {
        self.first.get(node as usize).copied().unwrap_or_default()
    }
}

/// The keywords the compiler honours; everything not here (or in [`IGNORED`]) is refused.
const HANDLED: &[&str] = &[
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "allOf",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

/// Keywords that carry no constraint on the value.
const IGNORED: &[&str] = &[
    "description",
    "title",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "definitions",
    "deprecated",
    "readOnly",
    "writeOnly",
];

fn ignored(key: &str) -> bool {
    IGNORED.contains(&key)
}

struct Compiler<'a> {
    out: Compiled,
    root: Option<&'a Value>,
    memo: HashMap<String, u32>,
    strict: bool,
}

impl<'a> Compiler<'a> {
    fn new() -> Compiler<'a> {
        let mut out = Compiled {
            nodes: Vec::new(),
            lits: Vec::new(),
            objects: Vec::new(),
            arrays: Vec::new(),
            strings: Vec::new(),
            numbers: Vec::new(),
            any: 0,
            any_string: 0,
            root: 0,
            first: Vec::new(),
            min_len: Vec::new(),
        };
        // `Any` is an alternative of every value kind; its pieces are built once.
        out.push(Node::Any);
        let any = out.any;
        let obj = {
            let id = out.objects.len() as u32;
            out.objects.push(ObjectSchema {
                properties: Vec::new(),
                additional: Additional::Any,
            });
            out.push(Node::Obj(id))
        };
        let arr = {
            let id = out.arrays.len() as u32;
            out.arrays.push(ArraySchema {
                prefix: Vec::new(),
                items: any,
                min: 0,
                max: u32::MAX,
            });
            out.push(Node::Arr(id))
        };
        let string_schema = {
            let id = out.strings.len() as u32;
            out.strings.push(StringSchema {
                min: 0,
                max: u32::MAX,
            });
            id
        };
        let string = out.push(Node::Str(string_schema));
        let number = {
            let id = out.numbers.len() as u32;
            out.numbers.push(NumberSchema {
                integer: false,
                sign: Sign::Any,
            });
            out.push(Node::Num(id))
        };
        let t = out.lit(b"true".to_vec());
        let f = out.lit(b"false".to_vec());
        let n = out.lit(b"null".to_vec());
        let any_alt = out.push(Node::Alt(vec![obj, arr, string, number, t, f, n]));
        out.any = any_alt;
        out.any_string = string_schema;
        Compiler {
            out,
            root: None,
            memo: HashMap::new(),
            strict: false,
        }
    }

    fn ast(&mut self, ast: &'a Ast) -> Result<u32, SchemaError> {
        match ast {
            Ast::Lit(bytes) => Ok(self.out.lit(bytes.clone())),
            Ast::Json(schema) => self.json_part(&schema.value, schema.strict),
            Ast::Seq(parts) => {
                let mut ids = Vec::with_capacity(parts.len());
                for p in parts {
                    ids.push(self.ast(p)?);
                }
                Ok(self.out.push(Node::Seq(ids)))
            }
            Ast::Alt(alts) => {
                let mut ids = Vec::with_capacity(alts.len());
                for a in alts {
                    ids.push(self.ast(a)?);
                }
                Ok(self.out.push(Node::Alt(ids)))
            }
            Ast::Repeat(inner, min, max) => {
                if max < min {
                    return Err(SchemaError::new(
                        "repeat",
                        "#",
                        format!("max ({max}) is less than min ({min})"),
                    ));
                }
                let id = self.ast(inner)?;
                Ok(self.out.push(Node::Repeat {
                    inner: id,
                    min: *min,
                    max: *max,
                }))
            }
            Ast::JsonObject => {
                let id = self.out.objects.len() as u32;
                self.out.objects.push(ObjectSchema {
                    properties: Vec::new(),
                    additional: Additional::Any,
                });
                Ok(self.out.push(Node::Obj(id)))
            }
        }
    }

    /// Compile one schema document; its `$ref`s resolve inside this document.
    fn json_part(&mut self, value: &'a Value, strict: bool) -> Result<u32, SchemaError> {
        let saved_root = self.root.replace(value);
        let saved_memo = std::mem::take(&mut self.memo);
        let saved_strict = std::mem::replace(&mut self.strict, strict);
        let result = self.compile_ptr("#");
        self.root = saved_root;
        self.memo = saved_memo;
        self.strict = saved_strict;
        result
    }

    fn root(&self) -> &'a Value {
        self.root.expect("a schema compile installs its root")
    }

    fn compile_ptr(&mut self, ptr: &str) -> Result<u32, SchemaError> {
        if let Some(&id) = self.memo.get(ptr) {
            return Ok(id);
        }
        let Some(value) = resolve_pointer(self.root(), ptr) else {
            return Err(SchemaError::new(
                "$ref",
                ptr,
                "the reference does not resolve inside this schema document",
            ));
        };
        let id = self.out.nodes.len() as u32;
        // Reserve the id first, so a recursive reference to this pointer finds it.
        self.out.nodes.push(Node::Lit(u32::MAX));
        self.memo.insert(ptr.to_string(), id);
        let node = self.compile_value(value, ptr)?;
        // The reserved id is what a recursive reference points at. Alias the compiled
        // node with an `Alt` of one: the node may be a placeholder itself (a `$ref` to a
        // pointer still being compiled), and copying it now would freeze the placeholder.
        // A pointer whose schema is only `{"$ref": "#"}` refers to itself and matches
        // nothing; that empty alternative is reported by `analyse` rather than looping.
        self.out.nodes[id as usize] = if node == id {
            Node::Alt(Vec::new())
        } else {
            Node::Alt(vec![node])
        };
        Ok(id)
    }

    fn compile_value(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        let Value::Obj(entries) = value else {
            return Err(SchemaError::new(
                "type",
                ptr,
                "a schema must be a JSON object",
            ));
        };
        for (key, _) in entries {
            if !HANDLED.contains(&key.as_str()) && !ignored(key) {
                return Err(SchemaError::new(
                    key,
                    ptr,
                    "it would restrict the reply and this backend cannot honour it",
                ));
            }
        }
        if let Some(reference) = value.get("$ref") {
            let Some(target) = reference.as_str() else {
                return Err(SchemaError::new("$ref", ptr, "a $ref must be a string"));
            };
            if !target.starts_with('#') {
                return Err(SchemaError::new(
                    "$ref",
                    ptr,
                    format!(
                        "only a local reference (starting with '#') is supported, not {target:?}"
                    ),
                ));
            }
            for (key, _) in entries {
                if key != "$ref" && !ignored(key) {
                    return Err(SchemaError::new(
                        key,
                        ptr,
                        "it cannot be combined with $ref",
                    ));
                }
            }
            return self.compile_ptr(target);
        }
        if value.get("allOf").is_some() {
            return self.compile_all_of(value, ptr);
        }
        if value.get("anyOf").is_some() || value.get("oneOf").is_some() {
            let key = if value.get("anyOf").is_some() {
                "anyOf"
            } else {
                "oneOf"
            };
            for (k, _) in entries {
                if k != key && !ignored(k) {
                    return Err(SchemaError::new(
                        k,
                        ptr,
                        format!(
                            "it cannot be combined with {key} (a conjunction is not supported)"
                        ),
                    ));
                }
            }
            let Some(items) = value.get(key).and_then(Value::as_array) else {
                return Err(SchemaError::new(key, ptr, "it must be an array of schemas"));
            };
            if items.is_empty() {
                return Err(SchemaError::new(key, ptr, "it must not be empty"));
            }
            let mut ids = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let child = format!("{ptr}/{key}/{i}");
                ids.push(self.compile_value(item, &child)?);
            }
            return Ok(self.out.push(Node::Alt(ids)));
        }
        if value.get("const").is_some() || value.get("enum").is_some() {
            return self.compile_enum(value, ptr);
        }
        match value.get("type") {
            Some(Value::Str(t)) => self.compile_typed(value, t, ptr),
            Some(Value::Arr(types)) => {
                let mut ids = Vec::with_capacity(types.len());
                for (i, t) in types.iter().enumerate() {
                    let Some(t) = t.as_str() else {
                        return Err(SchemaError::new(
                            "type",
                            format!("{ptr}/type/{i}"),
                            "a type must be a string",
                        ));
                    };
                    ids.push(self.compile_typed(value, t, ptr)?);
                }
                Ok(self.out.push(Node::Alt(ids)))
            }
            Some(_) => Err(SchemaError::new(
                "type",
                ptr,
                "type must be a string or an array of strings",
            )),
            None => self.compile_untyped(value, ptr),
        }
    }

    fn compile_untyped(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        for key in [
            "properties",
            "required",
            "additionalProperties",
            "items",
            "prefixItems",
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
        ] {
            if value.get(key).is_some() {
                return Err(SchemaError::new(
                    "type",
                    ptr,
                    format!(
                        "\"type\" is required when {key:?} is present (it constrains only one JSON type)"
                    ),
                ));
            }
        }
        Ok(self.out.any)
    }

    fn compile_all_of(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        let Some(items) = value.get("allOf").and_then(Value::as_array) else {
            return Err(SchemaError::new(
                "allOf",
                ptr,
                "it must be an array of schemas",
            ));
        };
        if items.is_empty() {
            return Err(SchemaError::new("allOf", ptr, "it must not be empty"));
        }
        for (k, _) in value.as_object().unwrap_or(&[]) {
            if k != "allOf" && !ignored(k) {
                return Err(SchemaError::new(k, ptr, "it cannot be combined with allOf"));
            }
        }
        if items.len() == 1 {
            return self.compile_value(&items[0], &format!("{ptr}/allOf/0"));
        }
        // A merge of unambiguous object schemas: properties are unioned in order, a
        // duplicate name must carry the same schema, `required` is the union, and an
        // additionalProperties that conflicts is refused.
        let mut properties: Vec<(String, Value)> = Vec::new();
        let mut required: Vec<String> = Vec::new();
        let mut additional: Option<Value> = None;
        for (i, item) in items.iter().enumerate() {
            let item_ptr = format!("{ptr}/allOf/{i}");
            let item = self.deref_owned(item, &item_ptr)?;
            let Value::Obj(_) = &item else {
                return Err(SchemaError::new(
                    "allOf",
                    &item_ptr,
                    "only object schemas can be merged by allOf",
                ));
            };
            match item.get("type") {
                Some(Value::Str(t)) if t == "object" => {}
                None if item.get("properties").is_some() => {}
                _ => {
                    return Err(SchemaError::new(
                        "allOf",
                        &item_ptr,
                        "only object schemas can be merged by allOf",
                    ))
                }
            }
            if let Some(Value::Obj(props)) = item.get("properties") {
                for (name, schema) in props {
                    if let Some((_, prev)) = properties.iter().find(|(n, _)| n == name) {
                        if prev != schema {
                            return Err(SchemaError::new(
                                "allOf",
                                &item_ptr,
                                format!("the property {name:?} has two different schemas"),
                            ));
                        }
                    } else {
                        properties.push((name.clone(), schema.clone()));
                    }
                }
            } else if item.get("properties").is_some() {
                return Err(SchemaError::new(
                    "properties",
                    &item_ptr,
                    "properties must be an object",
                ));
            }
            if let Some(req) = item.get("required") {
                let Some(req) = req.as_array() else {
                    return Err(SchemaError::new(
                        "required",
                        &item_ptr,
                        "required must be an array of strings",
                    ));
                };
                for r in req {
                    let Some(r) = r.as_str() else {
                        return Err(SchemaError::new(
                            "required",
                            &item_ptr,
                            "required must be an array of strings",
                        ));
                    };
                    if !required.iter().any(|x| x == r) {
                        required.push(r.to_string());
                    }
                }
            }
            if let Some(add) = item.get("additionalProperties") {
                match &additional {
                    Some(prev) if prev != add => {
                        return Err(SchemaError::new(
                            "allOf",
                            &item_ptr,
                            "the elements disagree on additionalProperties",
                        ))
                    }
                    None => additional = Some(add.clone()),
                    _ => {}
                }
            }
        }
        let mut merged = vec![("type".to_string(), Value::Str("object".to_string()))];
        merged.push(("properties".to_string(), Value::Obj(properties)));
        if !required.is_empty() {
            merged.push((
                "required".to_string(),
                Value::Arr(required.into_iter().map(Value::Str).collect()),
            ));
        }
        if let Some(add) = additional {
            merged.push(("additionalProperties".to_string(), add));
        }
        self.compile_value(&Value::Obj(merged), ptr)
    }

    /// Follow a chain of `$ref`s to a schema body.
    fn deref_owned(&self, value: &Value, ptr: &str) -> Result<Value, SchemaError> {
        let mut current = value.clone();
        for _ in 0..64 {
            let Some(reference) = current.get("$ref") else {
                return Ok(current);
            };
            let Some(target) = reference.as_str() else {
                return Err(SchemaError::new("$ref", ptr, "a $ref must be a string"));
            };
            let Some(next) = resolve_pointer(self.root(), target) else {
                return Err(SchemaError::new(
                    "$ref",
                    ptr,
                    format!("the reference {target:?} does not resolve"),
                ));
            };
            current = next.clone();
        }
        Err(SchemaError::new("$ref", ptr, "a $ref chain is too deep"))
    }

    fn compile_enum(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        for (key, _) in value.as_object().unwrap_or(&[]) {
            if key != "enum" && key != "const" && key != "type" && !ignored(key) {
                return Err(SchemaError::new(
                    key,
                    ptr,
                    "it cannot be combined with enum/const (the values would decide the language)",
                ));
            }
        }
        let mut values: Vec<&Value> = Vec::new();
        if let Some(c) = value.get("const") {
            values.push(c);
        } else {
            let Some(Value::Arr(items)) = value.get("enum") else {
                return Err(SchemaError::new("enum", ptr, "enum must be an array"));
            };
            if items.is_empty() {
                return Err(SchemaError::new("enum", ptr, "enum must not be empty"));
            }
            values.extend(items.iter());
        }
        let types: Vec<&str> = match value.get("type") {
            Some(Value::Str(t)) => vec![t.as_str()],
            Some(Value::Arr(ts)) => ts.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        let mut ids = Vec::new();
        for v in values {
            if types.is_empty() || types.iter().any(|t| type_matches(t, v)) {
                let bytes = v.to_jinja_json().into_bytes();
                ids.push(self.out.lit(bytes));
            }
        }
        if ids.is_empty() {
            return Err(SchemaError::new(
                "enum",
                ptr,
                "no enum value has the declared type",
            ));
        }
        Ok(if ids.len() == 1 {
            ids[0]
        } else {
            self.out.push(Node::Alt(ids))
        })
    }

    fn compile_typed(&mut self, value: &Value, kind: &str, ptr: &str) -> Result<u32, SchemaError> {
        match kind {
            "object" => self.compile_object(value, ptr),
            "array" => self.compile_array(value, ptr),
            "string" => self.compile_string(value, ptr),
            "integer" => self.compile_number(value, true, ptr),
            "number" => self.compile_number(value, false, ptr),
            "boolean" => {
                let t = self.out.lit(b"true".to_vec());
                let f = self.out.lit(b"false".to_vec());
                Ok(self.out.push(Node::Alt(vec![t, f])))
            }
            "null" => Ok(self.out.lit(b"null".to_vec())),
            other => Err(SchemaError::new(
                "type",
                ptr,
                format!("{other:?} is not a JSON Schema type"),
            )),
        }
    }

    fn compile_object(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        let mut props: Vec<Property> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        match value.get("properties") {
            None | Some(Value::Null) => {}
            Some(Value::Obj(entries)) => {
                for (name, schema) in entries {
                    let child = format!("{ptr}/properties/{}", escape_pointer(name));
                    let node = self.compile_value(schema, &child)?;
                    let key = self.out.lit_id(json_string(name).into_bytes());
                    props.push(Property {
                        key,
                        value: node,
                        required: false,
                    });
                    names.push(name.clone());
                }
            }
            Some(_) => {
                return Err(SchemaError::new(
                    "properties",
                    ptr,
                    "properties must be an object",
                ))
            }
        }
        let mut required: Vec<String> = Vec::new();
        match value.get("required") {
            None | Some(Value::Null) => {}
            Some(Value::Arr(items)) => {
                for item in items {
                    let Some(name) = item.as_str() else {
                        return Err(SchemaError::new(
                            "required",
                            ptr,
                            "required must be an array of strings",
                        ));
                    };
                    required.push(name.to_string());
                }
            }
            Some(_) => {
                return Err(SchemaError::new(
                    "required",
                    ptr,
                    "required must be an array of strings",
                ))
            }
        }
        for name in &required {
            if !names.iter().any(|n| n == name) {
                return Err(SchemaError::new(
                    "required",
                    ptr,
                    format!("the required property {name:?} is not declared in properties"),
                ));
            }
        }
        for (p, name) in props.iter_mut().zip(&names) {
            p.required = self.strict || required.iter().any(|r| r == name);
        }
        let additional = if self.strict {
            Additional::None
        } else {
            match value.get("additionalProperties") {
                None | Some(Value::Null) => Additional::Any,
                Some(Value::Bool(false)) => Additional::None,
                Some(Value::Bool(true)) => Additional::Any,
                Some(v @ Value::Obj(_)) => {
                    let child = format!("{ptr}/additionalProperties");
                    Additional::Node(self.compile_value(v, &child)?)
                }
                Some(_) => {
                    return Err(SchemaError::new(
                        "additionalProperties",
                        ptr,
                        "additionalProperties must be false, true or a schema",
                    ))
                }
            }
        };
        let id = self.out.objects.len() as u32;
        self.out.objects.push(ObjectSchema {
            properties: props,
            additional,
        });
        Ok(self.out.push(Node::Obj(id)))
    }

    fn compile_array(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        let mut prefix = Vec::new();
        match value.get("prefixItems") {
            None | Some(Value::Null) => {}
            Some(Value::Arr(items)) => {
                for (i, item) in items.iter().enumerate() {
                    prefix.push(self.compile_value(item, &format!("{ptr}/prefixItems/{i}"))?);
                }
            }
            Some(_) => {
                return Err(SchemaError::new(
                    "prefixItems",
                    ptr,
                    "prefixItems must be an array of schemas",
                ))
            }
        }
        let items = match value.get("items") {
            None | Some(Value::Null) => self.out.any,
            Some(v @ Value::Obj(_)) => self.compile_value(v, &format!("{ptr}/items"))?,
            Some(_) => return Err(SchemaError::new("items", ptr, "items must be a schema")),
        };
        let min = match value.get("minItems") {
            None | Some(Value::Null) => 0,
            Some(v) => integer_keyword(v, "minItems", ptr)?,
        };
        let max = match value.get("maxItems") {
            None | Some(Value::Null) => u32::MAX,
            Some(v) => integer_keyword(v, "maxItems", ptr)?,
        };
        if min > max {
            return Err(SchemaError::new(
                "minItems",
                ptr,
                "minItems is greater than maxItems",
            ));
        }
        let id = self.out.arrays.len() as u32;
        self.out.arrays.push(ArraySchema {
            prefix,
            items,
            min,
            max,
        });
        Ok(self.out.push(Node::Arr(id)))
    }

    fn compile_string(&mut self, value: &Value, ptr: &str) -> Result<u32, SchemaError> {
        let min = match value.get("minLength") {
            None | Some(Value::Null) => 0,
            Some(v) => integer_keyword(v, "minLength", ptr)?,
        };
        let max = match value.get("maxLength") {
            None | Some(Value::Null) => u32::MAX,
            Some(v) => integer_keyword(v, "maxLength", ptr)?,
        };
        if min > max {
            return Err(SchemaError::new(
                "minLength",
                ptr,
                "minLength is greater than maxLength",
            ));
        }
        let id = self.out.strings.len() as u32;
        self.out.strings.push(StringSchema { min, max });
        Ok(self.out.push(Node::Str(id)))
    }

    fn compile_number(
        &mut self,
        value: &Value,
        integer: bool,
        ptr: &str,
    ) -> Result<u32, SchemaError> {
        let mut lo: Option<i128> = None;
        let mut hi: Option<i128> = None;
        for (key, exclusive, is_min) in [
            ("minimum", false, true),
            ("exclusiveMinimum", true, true),
            ("maximum", false, false),
            ("exclusiveMaximum", true, false),
        ] {
            let Some(v) = value.get(key) else {
                continue;
            };
            if !integer {
                return Err(SchemaError::new(
                    key,
                    ptr,
                    "numeric bounds are supported only for integers",
                ));
            }
            let Some(bound) = bound_f64(v) else {
                return Err(SchemaError::new(
                    key,
                    ptr,
                    "the bound must be a finite number",
                ));
            };
            // inclusive lower = ceil(bound); exclusive lower = floor(bound)+1.
            let adjusted = if is_min {
                if exclusive {
                    floor_i128(bound).saturating_add(1)
                } else {
                    ceil_i128(bound)
                }
            } else if exclusive {
                ceil_i128(bound).saturating_sub(1)
            } else {
                floor_i128(bound)
            };
            if is_min {
                lo = Some(adjusted);
            } else {
                hi = Some(adjusted);
            }
        }
        match (lo, hi) {
            (None, None) => {
                let id = self.out.numbers.len() as u32;
                self.out.numbers.push(NumberSchema {
                    integer,
                    sign: Sign::Any,
                });
                Ok(self.out.push(Node::Num(id)))
            }
            (Some(lo), Some(hi)) => {
                if hi < lo {
                    return Err(SchemaError::new(
                        "minimum",
                        ptr,
                        "the range is empty (minimum is greater than maximum)",
                    ));
                }
                let count = hi - lo + 1;
                if count > MAX_FINITE_INT_VALUES {
                    return Err(SchemaError::new(
                        "minimum",
                        ptr,
                        format!(
                            "the integer range holds {count} values; at most {MAX_FINITE_INT_VALUES} are supported"
                        ),
                    ));
                }
                let mut ids = Vec::with_capacity(count as usize);
                let mut v = lo;
                while v <= hi {
                    ids.push(self.out.lit(v.to_string().into_bytes()));
                    v += 1;
                }
                Ok(self.out.push(Node::Alt(ids)))
            }
            (Some(lo), None) => {
                if lo == 0 {
                    let id = self.out.numbers.len() as u32;
                    self.out.numbers.push(NumberSchema {
                        integer,
                        sign: Sign::NonNegative,
                    });
                    Ok(self.out.push(Node::Num(id)))
                } else {
                    Err(SchemaError::new(
                        "minimum",
                        ptr,
                        "a single lower bound is supported only as the sign constraint minimum 0",
                    ))
                }
            }
            (None, Some(hi)) => {
                if hi == 0 {
                    let id = self.out.numbers.len() as u32;
                    self.out.numbers.push(NumberSchema {
                        integer,
                        sign: Sign::NonPositive,
                    });
                    Ok(self.out.push(Node::Num(id)))
                } else {
                    Err(SchemaError::new(
                        "maximum",
                        ptr,
                        "a single upper bound is supported only as the sign constraint maximum 0",
                    ))
                }
            }
        }
    }
}

fn type_matches(kind: &str, value: &Value) -> bool {
    matches!(
        (kind, value),
        ("object", Value::Obj(_))
            | ("array", Value::Arr(_))
            | ("string", Value::Str(_))
            | ("number", Value::Int(_) | Value::Float(_))
            | ("integer", Value::Int(_))
            | ("boolean", Value::Bool(_))
            | ("null", Value::Null)
    )
}

fn integer_keyword(value: &Value, key: &str, ptr: &str) -> Result<u32, SchemaError> {
    match value {
        Value::Int(i) if *i >= 0 => Ok((*i).min(u32::MAX as i64) as u32),
        Value::Int(_) => Err(SchemaError::new(key, ptr, "it must not be negative")),
        _ => Err(SchemaError::new(
            key,
            ptr,
            "it must be a non-negative integer",
        )),
    }
}

fn bound_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int(i) => Some(*i as f64),
        Value::Float(f) if f.is_finite() => Some(*f),
        _ => None,
    }
}

fn floor_i128(v: f64) -> i128 {
    if v <= i128::MIN as f64 {
        i128::MIN / 2
    } else if v >= i128::MAX as f64 {
        i128::MAX / 2
    } else {
        v.floor() as i128
    }
}

fn ceil_i128(v: f64) -> i128 {
    if v <= i128::MIN as f64 {
        i128::MIN / 2
    } else if v >= i128::MAX as f64 {
        i128::MAX / 2
    } else {
        v.ceil() as i128
    }
}

/// A JSON Pointer with `~0`/`~1` escaping, as `format!("{ptr}/properties/{name}")` needs.
fn escape_pointer(name: &str) -> String {
    name.replace('~', "~0").replace('/', "~1")
}

fn unescape_pointer(part: &str) -> String {
    part.replace("~1", "/").replace("~0", "~")
}

/// Resolve a local JSON pointer (`#`, `#/a/b/0`) inside `root`.
fn resolve_pointer<'a>(root: &'a Value, ptr: &str) -> Option<&'a Value> {
    if ptr == "#" {
        return Some(root);
    }
    let rest = ptr.strip_prefix("#/")?;
    let mut current = root;
    for raw in rest.split('/') {
        let part = unescape_pointer(raw);
        current = match current {
            Value::Obj(entries) => entries.iter().find(|(k, _)| *k == part).map(|(_, v)| v)?,
            Value::Arr(items) => {
                let i: usize = part.parse().ok()?;
                items.get(i)?
            }
            _ => return None,
        };
    }
    Some(current)
}

/// A JSON string as JSON text: the minimal escaping RFC 8259 requires.
pub(crate) fn json_string(s: &str) -> String {
    Value::Str(s.to_string()).to_jinja_json()
}
