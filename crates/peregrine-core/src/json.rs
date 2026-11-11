//! A small JSON value that keeps the order of object keys, and the JSON text llama.cpp's
//! Jinja engine writes for it.
//!
//! The System One prompts (`systemone.rs`) put a request's `state`, instructions and option
//! descriptions into the prompt: a string as it is, anything else as JSON text. The text has to be
//! the one llama.cpp v0.9.0 produces, byte for byte, or the model reads a different prompt. That
//! pins two things an ordinary JSON library does not promise:
//!
//! - **Key order.** llama.cpp parses a request into `nlohmann::ordered_json` and its Jinja
//!   `tojson` writes object keys in that order (`common/jinja/value.cpp`,
//!   `value_to_json_internal`: "need to keep exact order"). `serde_json` without `preserve_order`
//!   sorts them.
//! - **Number and string spelling.** An integer is written as an `i64`; any other number is a
//!   `double` written the way Python's `repr` writes it (`float_repr` in the same file: the
//!   shortest digits that read back, fixed notation from `1e-4` up to `1e16`, `d.ddde+XX`
//!   outside). Strings escape `"`, `\`, the five short escapes and other control characters as
//!   `\u00xx`; everything else, non-ASCII included, is kept.
//!
//! The parser accepts standard JSON (RFC 8259). No dependency: the engine crate has none for JSON
//! and does not gain one for this.

use std::fmt::Write as _;

/// Arrays and objects nested deeper than this are refused, as llama.cpp's Jinja does
/// (`JINJA_RECURSIVE_LIMIT`), so a hostile request cannot overflow the stack.
pub const MAX_DEPTH: usize = 128;

/// A JSON value. Objects keep the order their keys were given in.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A number written without a fraction or exponent that fits an integer.
    Int(i64),
    /// Any other number.
    Float(f64),
    Str(String),
    Arr(Vec<Value>),
    /// Key–value pairs in insertion order. A repeated key keeps its first position and its last
    /// value, as `nlohmann::ordered_json`'s parser leaves it.
    Obj(Vec<(String, Value)>),
}

impl Value {
    /// Parse one JSON document. Trailing non-whitespace is an error.
    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("unexpected text after the JSON value"));
        }
        Ok(v)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Obj(o) => Some(o),
            _ => None,
        }
    }

    /// The value under `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Jinja's truth value as llama.cpp computes it (`as_bool` in `common/jinja/value.h`): null,
    /// `false`, zero, the empty string, the empty array and the empty object are false.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Arr(a) => !a.is_empty(),
            Value::Obj(o) => !o.is_empty(),
        }
    }

    /// The JSON text llama.cpp's Jinja `tojson` writes with its defaults: `", "` between items,
    /// `": "` after a key, no indentation, non-ASCII kept, keys in order.
    pub fn to_jinja_json(&self) -> String {
        let mut out = String::new();
        self.write_jinja_json(&mut out);
        out
    }

    fn write_jinja_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Value::Float(f) => out.push_str(&float_json(*f)),
            Value::Str(s) => write_string(s, out),
            Value::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_jinja_json(out);
                }
                out.push(']');
            }
            Value::Obj(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_string(k, out);
                    out.push_str(": ");
                    v.write_jinja_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// A string as llama.cpp's `value_to_json_internal` writes it.
fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float as llama.cpp's `float_json` writes it: `NaN` and `Infinity` (never produced by a JSON
/// parse, kept for completeness), else [`float_repr`].
fn float_json(v: f64) -> String {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v < 0.0 { "-Infinity" } else { "Infinity" }.into()
    } else {
        float_repr(v)
    }
}

/// A finite float as Python's `repr` writes it, which is what llama.cpp's `float_repr` computes:
/// the shortest digits that read back as the same double; fixed notation when the decimal point
/// falls within `(-4, 16]` digits of the first digit, with `.0` after an integral value; otherwise
/// `d.ddde+XX` with a sign and at least two exponent digits.
pub fn float_repr(v: f64) -> String {
    let mag = v.abs();
    // Rust's `{:e}` prints the shortest digits that read back, as `d.ddd` + `e` + exponent.
    let sci = format!("{mag:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp10: i32 = exp.parse().expect("a decimal exponent");
    let mut digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let n = digits.len() as i32;
    let point = exp10 + 1; // digits before the decimal point
    let mut out = String::new();
    if v.is_sign_negative() {
        out.push('-');
    }
    if point > -4 && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            for _ in 0..-point {
                out.push('0');
            }
            out.push_str(&digits);
        } else if point >= n {
            out.push_str(&digits);
            for _ in 0..point - n {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exp10 < 0 { '-' } else { '+' },
            exp10.abs()
        );
    }
    out
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("json: {what} at byte {}", self.i)
    }

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

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        match self.b.get(self.i) {
            None => Err(self.err("unexpected end of input")),
            Some(b'n') if self.eat("null") => Ok(Value::Null),
            Some(b't') if self.eat("true") => Ok(Value::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Value::Bool(false)),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b'[') => {
                if depth >= MAX_DEPTH {
                    return Err(self.err("nested too deep"));
                }
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Arr(items));
                }
                loop {
                    self.ws();
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Value::Arr(items));
                        }
                        _ => return Err(self.err("expected ',' or ']'")),
                    }
                }
            }
            Some(b'{') => {
                if depth >= MAX_DEPTH {
                    return Err(self.err("nested too deep"));
                }
                self.i += 1;
                let mut pairs: Vec<(String, Value)> = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Obj(pairs));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(self.err("expected a string key"));
                    }
                    let key = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(self.err("expected ':'"));
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    match pairs.iter_mut().find(|(k, _)| *k == key) {
                        Some(slot) => slot.1 = v,
                        None => pairs.push((key, v)),
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Value::Obj(pairs));
                        }
                        _ => return Err(self.err("expected ',' or '}'")),
                    }
                }
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<Value, String> {
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
            _ => return Err(self.err("invalid number")),
        }
        let mut integral = true;
        if self.b.get(self.i) == Some(&b'.') {
            integral = false;
            self.i += 1;
            if !matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number"));
            }
            while let Some(b'0'..=b'9') = self.b.get(self.i) {
                self.i += 1;
            }
        }
        if let Some(b'e' | b'E') = self.b.get(self.i) {
            integral = false;
            self.i += 1;
            if let Some(b'+' | b'-') = self.b.get(self.i) {
                self.i += 1;
            }
            if !matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number"));
            }
            while let Some(b'0'..=b'9') = self.b.get(self.i) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII digits");
        if integral {
            // nlohmann keeps an integer that fits 64 bits (signed, or unsigned when it is not
            // negative) and llama.cpp reads either as an `int64_t`; a longer one becomes a double.
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Value::Int(i));
            }
            if let Ok(u) = text.parse::<u64>() {
                return Ok(Value::Int(u as i64));
            }
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| self.err("invalid number"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| self.err("invalid \\u escape"))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            // The input is a `&str`, and the scan stops only at ASCII bytes, so the run is UTF-8.
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).expect("a str slice"));
            match self.b.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| self.err("unterminated escape"))?;
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
                                    return Err(self.err("unpaired surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("unpaired surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(
                                char::from_u32(cp).ok_or_else(|| self.err("unpaired surrogate"))?,
                            );
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                Some(_) => return Err(self.err("control character in a string")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_keep_their_key_order() {
        let v =
            Value::parse(r#"{"zeta": 1, "alpha": [true, null], "mid": {"b": 2, "a": 1}}"#).unwrap();
        assert_eq!(
            v.to_jinja_json(),
            r#"{"zeta": 1, "alpha": [true, null], "mid": {"b": 2, "a": 1}}"#
        );
    }

    #[test]
    fn a_repeated_key_keeps_its_place_and_takes_the_last_value() {
        let v = Value::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(v.to_jinja_json(), r#"{"a": 3, "b": 2}"#);
    }

    #[test]
    fn strings_escape_what_llama_cpp_escapes() {
        let v = Value::parse(
            r#""q\" b\\ n\n t\t r\r b\b f\f u\u0001 d\u007f é \u00e9 😀 \ud83d\ude00 /\/""#,
        )
        .unwrap();
        assert_eq!(
            v.to_jinja_json(),
            "\"q\\\" b\\\\ n\\n t\\t r\\r b\\b f\\f u\\u0001 d\u{7f} é é 😀 😀 //\""
        );
    }

    #[test]
    fn numbers_are_written_as_python_repr_writes_them() {
        for (text, want) in [
            ("0", "0"),
            ("-0", "0"),
            ("42", "42"),
            ("-7", "-7"),
            ("9223372036854775807", "9223372036854775807"),
            ("18446744073709551615", "-1"),
            ("1.0", "1.0"),
            ("-0.0", "-0.0"),
            ("1.50", "1.5"),
            ("0.1", "0.1"),
            ("1e2", "100.0"),
            ("1E-4", "0.0001"),
            ("1e-5", "1e-05"),
            ("1.5e-7", "1.5e-07"),
            ("1e15", "1000000000000000.0"),
            ("1e16", "1e+16"),
            ("1.25e16", "1.25e+16"),
            ("123456789012345678901234567890", "1.2345678901234568e+29"),
            ("3.141592653589793", "3.141592653589793"),
            ("0.30000000000000004", "0.30000000000000004"),
            ("1e300", "1e+300"),
            ("5e-324", "5e-324"),
        ] {
            assert_eq!(Value::parse(text).unwrap().to_jinja_json(), want, "{text}");
        }
    }

    #[test]
    fn truth_follows_jinja() {
        for (text, want) in [
            ("null", false),
            ("false", false),
            ("true", true),
            ("0", false),
            ("3", true),
            ("0.0", false),
            ("0.5", true),
            ("\"\"", false),
            ("\" \"", true),
            ("[]", false),
            ("[0]", true),
            ("{}", false),
            ("{\"a\": null}", true),
        ] {
            assert_eq!(Value::parse(text).unwrap().is_truthy(), want, "{text}");
        }
    }

    #[test]
    fn malformed_input_is_refused() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "{'a': 1}",
            "01",
            "1.",
            "1e",
            "\"abc",
            "\"\\x\"",
            "\"\\ud83d\"",
            "nul",
            "1 2",
            "\"a\nb\"",
        ] {
            assert!(Value::parse(bad).is_err(), "{bad:?} parsed");
        }
        let deep = "[".repeat(MAX_DEPTH + 1) + &"]".repeat(MAX_DEPTH + 1);
        assert!(Value::parse(&deep).is_err());
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(Value::parse(&ok).is_ok());
    }
}
