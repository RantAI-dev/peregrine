//! Independent JSON parsing and schema validation for the constrained-decoding tests.
//!
//! It shares no code with `peregrine_core::constrain`: if the matcher and this validator
//! disagree, the property test fails. It covers exactly the supported schema subset.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number { value: f64, integral: bool },
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        b: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing text at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.b.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 128 {
            return Err("too deep".into());
        }
        match self.b.get(self.i) {
            None => Err("unterminated".into()),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                self.ws();
                let mut items = Vec::new();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    self.ws();
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Array(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                self.ws();
                let mut pairs = Vec::new();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Object(pairs));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(format!("expected a key at byte {}", self.i));
                    }
                    let key = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(format!("expected ':' at byte {}", self.i));
                    }
                    self.i += 1;
                    self.ws();
                    pairs.push((key, self.value(depth + 1)?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Object(pairs));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(&b) => Err(format!("unexpected byte {b} at {}", self.i)),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        match self.b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while let Some(b'0'..=b'9') = self.b.get(self.i) {
                    self.i += 1;
                }
            }
            _ => return Err(format!("bad number at byte {start}")),
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                return Err(format!("bad fraction at byte {}", self.i));
            }
            while let Some(b'0'..=b'9') = self.b.get(self.i) {
                self.i += 1;
            }
        }
        if let Some(b'e' | b'E') = self.b.get(self.i) {
            self.i += 1;
            if let Some(b'+' | b'-') = self.b.get(self.i) {
                self.i += 1;
            }
            if !matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                return Err(format!("bad exponent at byte {}", self.i));
            }
            while let Some(b'0'..=b'9') = self.b.get(self.i) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII");
        let integral = !text.contains(['.', 'e', 'E']);
        // A JSON number may be larger than an f64; the value saturates to infinity and
        // the syntax flag keeps "integer" meaningful.
        let value: f64 = text.parse().unwrap_or(f64::INFINITY);
        Ok(Json::Number { value, integral })
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| format!("bad \\u at byte {}", self.i))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).expect("input is str"));
            match self.b.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return Err("lone high surrogate".into());
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("lone high surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err("lone low surrogate".into());
                            } else {
                                hi
                            };
                            out.push(
                                char::from_u32(cp).ok_or_else(|| "bad code point".to_string())?,
                            );
                        }
                        _ => return Err("bad escape".into()),
                    }
                }
                Some(_) => return Err("control character in a string".into()),
            }
        }
    }
}

/// Validate `value` against `schema`, treating local `$ref`s as pointers into `root`.
pub fn validate(value: &Json, schema: &Json, root: &Json) -> Result<(), String> {
    let object = match schema {
        Json::Object(o) => o,
        Json::Bool(true) => return Ok(()),
        Json::Bool(false) => return Err("schema is false".into()),
        _ => return Err("schema is not an object".into()),
    };
    let get = |k: &str| object.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    if let Some(Json::String(target)) = get("$ref") {
        let mut current = resolve(root, target).ok_or_else(|| format!("bad $ref {target}"))?;
        for _ in 0..64 {
            let target = match current {
                Json::Object(pairs) => {
                    pairs
                        .iter()
                        .find(|(k, _)| k == "$ref")
                        .and_then(|(_, v)| match v {
                            Json::String(s) => Some(s.as_str()),
                            _ => None,
                        })
                }
                _ => None,
            };
            let Some(target) = target else { break };
            current = resolve(root, target).ok_or_else(|| format!("bad $ref {target}"))?;
        }
        return validate(value, current, root);
    }
    if let Some(all) = get("allOf") {
        let Json::Array(items) = all else {
            return Err("allOf".into());
        };
        for s in items {
            validate(value, s, root)?;
        }
    }
    if let Some(any) = get("anyOf") {
        let Json::Array(items) = any else {
            return Err("anyOf".into());
        };
        if !items.iter().any(|s| validate(value, s, root).is_ok()) {
            return Err("no anyOf alternative validates".into());
        }
    }
    if let Some(any) = get("oneOf") {
        let Json::Array(items) = any else {
            return Err("oneOf".into());
        };
        if !items.iter().any(|s| validate(value, s, root).is_ok()) {
            return Err("no oneOf alternative validates".into());
        }
    }
    if let Some(c) = get("const") {
        if !equal(value, c) {
            return Err("const".into());
        }
        return Ok(());
    }
    if let Some(Json::Array(items)) = get("enum") {
        if !items.iter().any(|v| equal(value, v)) {
            return Err("enum".into());
        }
        return Ok(());
    }
    if let Some(t) = get("type") {
        let ok = match t {
            Json::String(t) => type_ok(value, t),
            Json::Array(ts) => ts.iter().any(|t| match t {
                Json::String(t) => type_ok(value, t),
                _ => false,
            }),
            _ => false,
        };
        if !ok {
            return Err(format!("type: {value:?}"));
        }
    }
    match value {
        Json::Object(pairs) => {
            let strict = false;
            validate_object(pairs, object, root, strict)
        }
        Json::Array(items) => validate_array(items, object, root),
        Json::String(s) => {
            let chars = s.chars().count() as f64;
            if let Some(Json::Number { value: min, .. }) = get("minLength") {
                if chars < *min {
                    return Err("minLength".into());
                }
            }
            if let Some(Json::Number { value: max, .. }) = get("maxLength") {
                if chars > *max {
                    return Err("maxLength".into());
                }
            }
            Ok(())
        }
        Json::Number { value, integral } => validate_number(*value, *integral, object),
        _ => Ok(()),
    }
}

fn validate_object(
    pairs: &[(String, Json)],
    schema: &[(String, Json)],
    root: &Json,
    strict: bool,
) -> Result<(), String> {
    let get = |k: &str| schema.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    let properties = match get("properties") {
        Some(Json::Object(p)) => Some(p),
        None => None,
        Some(_) => return Err("properties".into()),
    };
    if let Some(Json::Array(req)) = get("required") {
        for r in req {
            let Json::String(r) = r else {
                return Err("required".into());
            };
            if !pairs.iter().any(|(k, _)| k == r) {
                return Err(format!("missing required {r:?}"));
            }
        }
    }
    if strict {
        if let Some(props) = properties {
            for (name, _) in props {
                if !pairs.iter().any(|(k, _)| k == name) {
                    return Err(format!("strict: missing {name:?}"));
                }
            }
        }
    }
    for (name, v) in pairs {
        let declared = properties.and_then(|p| p.iter().find(|(k, _)| k == name));
        match declared {
            Some((_, s)) => validate(v, s, root)?,
            None => match get("additionalProperties") {
                Some(Json::Bool(false)) | None if strict => {
                    return Err(format!("additional property {name:?}"))
                }
                Some(Json::Bool(false)) => return Err(format!("additional property {name:?}")),
                Some(Json::Bool(true)) | None => {}
                Some(s) => validate(v, s, root)?,
            },
        }
    }
    Ok(())
}

fn validate_array(items: &[Json], schema: &[(String, Json)], root: &Json) -> Result<(), String> {
    let get = |k: &str| schema.iter().find(|(n, _)| n == k).map(|(_, v)| v);
    if let Some(Json::Number { value: min, .. }) = get("minItems") {
        if (items.len() as f64) < *min {
            return Err("minItems".into());
        }
    }
    if let Some(Json::Number { value: max, .. }) = get("maxItems") {
        if (items.len() as f64) > *max {
            return Err("maxItems".into());
        }
    }
    let prefix = match get("prefixItems") {
        Some(Json::Array(p)) => p.as_slice(),
        _ => &[],
    };
    for (i, item) in items.iter().enumerate() {
        let s = prefix.get(i).or_else(|| get("items"));
        if let Some(s) = s {
            validate(item, s, root)?;
        }
    }
    Ok(())
}

fn validate_number(n: f64, integral: bool, schema: &[(String, Json)]) -> Result<(), String> {
    let get = |k: &str| schema.iter().find(|(name, _)| name == k).map(|(_, v)| v);
    if let Some(Json::String(t)) = get("type") {
        if t == "integer" && !integral && n.fract() != 0.0 {
            return Err("integer".into());
        }
    }
    for (key, exclusive, min) in [
        ("minimum", false, true),
        ("exclusiveMinimum", true, true),
        ("maximum", false, false),
        ("exclusiveMaximum", true, false),
    ] {
        if let Some(Json::Number { value: b, .. }) = get(key) {
            let ok = if min {
                if exclusive {
                    n > *b
                } else {
                    n >= *b
                }
            } else if exclusive {
                n < *b
            } else {
                n <= *b
            };
            if !ok {
                return Err(key.into());
            }
        }
    }
    Ok(())
}

fn type_ok(v: &Json, t: &str) -> bool {
    match (v, t) {
        (Json::Object(_), "object") => true,
        (Json::Array(_), "array") => true,
        (Json::String(_), "string") => true,
        (Json::Number { .. }, "number") => true,
        (Json::Number { value, integral }, "integer") => *integral || value.fract() == 0.0,
        (Json::Bool(_), "boolean") => true,
        (Json::Null, "null") => true,
        _ => false,
    }
}

fn equal(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Bool(x), Json::Bool(y)) => x == y,
        (Json::Number { value: x, .. }, Json::Number { value: y, .. }) => x == y,
        (Json::String(x), Json::String(y)) => x == y,
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| equal(a, b))
        }
        (Json::Object(x), Json::Object(y)) => {
            x.len() == y.len()
                && x.iter().all(|(k, v)| {
                    y.iter()
                        .find(|(k2, _)| k2 == k)
                        .is_some_and(|(_, v2)| equal(v, v2))
                })
        }
        _ => false,
    }
}

fn resolve<'a>(root: &'a Json, ptr: &str) -> Option<&'a Json> {
    if ptr == "#" {
        return Some(root);
    }
    let rest = ptr.strip_prefix("#/")?;
    let mut current = root;
    for part in rest.split('/') {
        let part = part.replace("~1", "/").replace("~0", "~");
        current = match current {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| *k == part).map(|(_, v)| v)?,
            Json::Array(items) => {
                let index: usize = part.parse().ok()?;
                items.get(index)?
            }
            _ => return None,
        };
    }
    Some(current)
}
