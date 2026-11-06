//! GGUF v3 file writer for test fixtures and synthetic models.

use peregrine_core::gguf::Format;
use std::fs::File;
use std::io::{self, BufWriter, Seek, Write};
use std::path::Path;

/// Metadata value types according to the GGUF specification.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
}

/// Metadata entry (key, value).
struct MetaEntry {
    key: String,
    val: MetaValue,
}

/// Tensor directory entry and payload.
struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    kind: Format,
    data: Vec<u8>,
}

/// Builder that writes GGUF v3 files.
pub struct GgufWriter {
    alignment: usize,
    meta: Vec<MetaEntry>,
    tensors: Vec<TensorEntry>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    /// Creates a new writer with default 32-byte alignment.
    pub fn new() -> Self {
        Self {
            alignment: 32,
            meta: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Sets the alignment for tensor data (must be a power of two).
    pub fn set_alignment(&mut self, alignment: usize) {
        assert!(alignment > 0 && alignment.is_power_of_two());
        self.alignment = alignment;
    }

    pub fn u32(&mut self, key: &str, v: u32) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::U32(v),
        });
    }

    pub fn i32(&mut self, key: &str, v: i32) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::I32(v),
        });
    }

    pub fn u64(&mut self, key: &str, v: u64) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::U64(v),
        });
    }

    pub fn f32(&mut self, key: &str, v: f32) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::F32(v),
        });
    }

    pub fn bool(&mut self, key: &str, v: bool) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::Bool(v),
        });
    }

    pub fn str(&mut self, key: &str, v: &str) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::Str(v.to_string()),
        });
    }

    pub fn arr_str<S: AsRef<str>>(&mut self, key: &str, v: &[S]) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::ArrStr(v.iter().map(|s| s.as_ref().to_string()).collect()),
        });
    }

    pub fn arr_i32(&mut self, key: &str, v: &[i32]) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::ArrI32(v.to_vec()),
        });
    }

    pub fn arr_f32(&mut self, key: &str, v: &[f32]) {
        self.meta.push(MetaEntry {
            key: key.to_string(),
            val: MetaValue::ArrF32(v.to_vec()),
        });
    }

    /// Adds a tensor to the model.
    pub fn add_tensor(&mut self, name: &str, dims: &[u64], kind: Format, data: Vec<u8>) {
        let numel: u64 = dims.iter().product();
        assert_eq!(
            numel % kind.block() as u64,
            0,
            "tensor {name} elements {numel} not a multiple of {} block {}",
            kind.label(),
            kind.block()
        );
        let expected_bytes = (numel / kind.block() as u64) * kind.block_bytes() as u64;
        assert_eq!(
            data.len() as u64,
            expected_bytes,
            "tensor {name} byte count mismatch: got {}, expected {expected_bytes}",
            data.len()
        );
        self.tensors.push(TensorEntry {
            name: name.to_string(),
            dims: dims.to_vec(),
            kind,
            data,
        });
    }

    /// Writes the GGUF file to disk.
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Header: magic "GGUF", version 3, n_tensors, n_kv
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        w.write_all(&(self.meta.len() as u64).to_le_bytes())?;

        // Metadata KV
        for m in &self.meta {
            write_string(&mut w, &m.key)?;
            write_meta_value(&mut w, &m.val)?;
        }

        // Tensor directory
        // Calculate offsets: each tensor starts at a multiple of `self.alignment` relative to `data_start`.
        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
        let mut curr_offset: u64 = 0;
        for t in &self.tensors {
            tensor_offsets.push(curr_offset);
            let len = t.data.len() as u64;
            let pad =
                (self.alignment as u64 - (len % self.alignment as u64)) % self.alignment as u64;
            curr_offset += len + pad;
        }

        for (t, &offset) in self.tensors.iter().zip(&tensor_offsets) {
            write_string(&mut w, &t.name)?;
            w.write_all(&(t.dims.len() as u32).to_le_bytes())?;
            for &d in &t.dims {
                w.write_all(&d.to_le_bytes())?;
            }
            w.write_all(&format_to_id(t.kind).to_le_bytes())?;
            w.write_all(&offset.to_le_bytes())?;
        }

        // Pad to alignment boundary before tensor data
        let pos = w.stream_position()? as usize;
        let pad = (self.alignment - (pos % self.alignment)) % self.alignment;
        if pad > 0 {
            let zeros = vec![0u8; pad];
            w.write_all(&zeros)?;
        }

        // Write tensor data
        for t in &self.tensors {
            w.write_all(&t.data)?;
            let pad = (self.alignment - (t.data.len() % self.alignment)) % self.alignment;
            if pad > 0 {
                let zeros = vec![0u8; pad];
                w.write_all(&zeros)?;
            }
        }

        w.flush()?;
        Ok(())
    }
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_meta_value<W: Write>(w: &mut W, val: &MetaValue) -> io::Result<()> {
    match val {
        MetaValue::U32(v) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::I32(v) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::U64(v) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::F32(v) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        MetaValue::Bool(v) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[if *v { 1 } else { 0 }])?;
        }
        MetaValue::Str(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            write_string(w, s)?;
        }
        MetaValue::ArrStr(arr) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&8u32.to_le_bytes())?; // element type string
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for s in arr {
                write_string(w, s)?;
            }
        }
        MetaValue::ArrI32(arr) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&5u32.to_le_bytes())?; // element type int32
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &item in arr {
                w.write_all(&item.to_le_bytes())?;
            }
        }
        MetaValue::ArrF32(arr) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&6u32.to_le_bytes())?; // element type float32
            w.write_all(&(arr.len() as u64).to_le_bytes())?;
            for &item in arr {
                w.write_all(&item.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn format_to_id(fmt: Format) -> u32 {
    match fmt {
        Format::F32 => 0,
        Format::F16 => 1,
        Format::Q4_0 => 2,
        Format::Q5_0 => 6,
        Format::Q8_0 => 8,
        Format::Q4_K => 12,
        Format::Q5_K => 13,
        Format::Q6_K => 14,
        Format::BF16 => 30,
        Format::PQ2_0 => 142,
        Format::PTQ1_0 => 143,
    }
}
