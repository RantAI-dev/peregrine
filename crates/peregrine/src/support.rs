//! Support tables generated from the code (roadmap S2.7).

use peregrine_kernels::format::Format;
use peregrine_kernels::table::kernels_for;
use std::fmt::Write as _;

/// What this build supports: architectures, weight formats, CPU tiers, tokenizers, and chat templates.
pub fn support(markdown: bool) -> String {
    if markdown {
        support_markdown()
    } else {
        support_text()
    }
}

/// Plain text summary of build support.
pub fn support_text() -> String {
    let mut out = String::new();

    let _ = writeln!(out, "architectures:");
    let mut arch_names = Vec::new();
    for fam in peregrine_models::FAMILIES {
        arch_names.extend(fam.names().iter().copied());
    }
    if arch_names.is_empty() {
        let _ = writeln!(out, "  (none)");
    } else {
        let _ = writeln!(out, "  {}", arch_names.join(", "));
    }

    let _ = writeln!(out, "weight formats:");
    let supported_fmts = Format::supported();
    if supported_fmts.is_empty() {
        let _ = writeln!(out, "  (none)");
    } else {
        for fmt in supported_fmts {
            let tiers = kernels_for(fmt);
            let tiers_str = if tiers.is_empty() {
                "none".to_string()
            } else {
                tiers
                    .iter()
                    .map(|t| t.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let _ = writeln!(out, "  {:<8} {tiers_str}", fmt.label());
        }
    }

    let _ = writeln!(out, "CPU tiers:");
    let _ = writeln!(out, "  scalar, avx2, avx512");

    let _ = writeln!(out, "tokenizers:");
    let _ = writeln!(
        out,
        "  GPT-2 byte-level BPE ({})",
        peregrine_core::tokenizer::PRE_TYPES.join(", ")
    );
    let _ = writeln!(out, "  SentencePiece (llama)");

    let _ = writeln!(out, "chat templates:");
    let _ = writeln!(out, "  {}", peregrine_core::template::TEMPLATES.join(", "));

    out
}

/// Markdown table rows for README.md.
pub fn support_markdown() -> String {
    let mut out = String::new();

    // 1. Architectures
    let mut arch_items = Vec::new();
    for fam in peregrine_models::FAMILIES {
        for &name in fam.names() {
            let label = match name {
                "llama" => "Llama",
                "qwen2" => "Qwen 2",
                "qwen3" => "Qwen 3",
                "qwen35" => "Qwen 3.5 / Bonsai 2 (Gated DeltaNet hybrid, MTP)",
                "gemma3" => "Gemma 3",
                other => other,
            };
            arch_items.push(label);
        }
    }
    let arch_str = arch_items.join(", ");

    // 2. Weight formats
    let fmt_labels: Vec<&str> = Format::supported().into_iter().map(|f| f.label()).collect();
    let fmt_str = fmt_labels.join(", ");

    // 3. CPUs
    let cpu_str = "x86-64 AVX2, AVX-512 (VNNI, GFNI, VBMI); others through scalar code";

    // 4. Tokenizers
    let tok_str = "GPT-2 byte-level BPE, SentencePiece (llama)";

    // 5. Chat templates
    let tmpl_str = "read from the model file and rendered exactly as llama.cpp renders them: \
ChatML, Llama 3, Qwen 3, Gemma, Mistral (v0, v1, v7), Llama 2 \
— or named with `--chat-template` for a file that declares none";

    let _ = writeln!(out, "| | Supported now |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| Architectures | {arch_str} |");
    let _ = writeln!(out, "| Weight formats | {fmt_str} |");
    let _ = writeln!(out, "| CPUs | {cpu_str} |");
    let _ = writeln!(out, "| Tokenizers | {tok_str} |");
    let _ = writeln!(out, "| Chat templates | {tmpl_str} |");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_text() {
        let text = support_text();
        assert!(text.contains("architectures:"));
        assert!(text.contains("weight formats:"));
        assert!(text.contains("CPU tiers:"));
        assert!(text.contains("tokenizers:"));
        assert!(text.contains("chat templates:"));

        #[cfg(feature = "arch-llama")]
        assert!(text.contains("llama"));
        #[cfg(feature = "fmt-kquant")]
        assert!(text.contains("Q4_K"));
    }

    #[test]
    fn test_support_markdown() {
        let md = support_markdown();
        assert!(md.contains("| | Supported now |"));
        assert!(md.contains("| Architectures |"));
        assert!(md.contains("| Weight formats |"));
        assert!(md.contains("| CPUs |"));
        assert!(md.contains("| Tokenizers |"));
        assert!(md.contains("| Chat templates |"));

        #[cfg(feature = "arch-llama")]
        assert!(md.contains("Llama"));
        #[cfg(feature = "fmt-kquant")]
        assert!(md.contains("Q4_K"));
    }
}
