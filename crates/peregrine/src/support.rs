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
    let _ = writeln!(
        out,
        "  scalar, avx2, avx-vnni, avx512, neon, neon-dotprod, neon-i8mm"
    );

    let _ = writeln!(out, "tokenizers:");
    let _ = writeln!(
        out,
        "  GPT-2 byte-level BPE ({})",
        peregrine_core::tokenizer::PRE_TYPES.join(", ")
    );
    let _ = writeln!(out, "  SentencePiece BPE (llama)");
    let _ = writeln!(out, "  WordPiece (bert)");
    let _ = writeln!(out, "  SentencePiece unigram (t5)");

    let _ = writeln!(out, "chat templates:");
    let _ = writeln!(out, "  {}", peregrine_core::template::TEMPLATES.join(", "));

    let _ = writeln!(out, "embeddings:");
    let _ = writeln!(
        out,
        "  causal decoders (last / mean pooling) and bidirectional encoders (mean / cls; EmbeddingGemma, and Mistral 3 / Nemotron 3.5 Embed), POST /v1/embeddings, peregrine embed"
    );

    let _ = writeln!(out, "rerank and classify:");
    let _ = writeln!(
        out,
        "  a bert file's classifier head (cls, cls.output, classifier.output_labels), POST /v1/rerank and POST /v1/classify, peregrine rerank"
    );

    let _ = writeln!(out, "decisions:");
    let _ = writeln!(
        out,
        "  native decision models (openjev), POST /v1/systemone, peregrine decide"
    );

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
                "gemma-embedding" => "Gemma 3 embedding (EmbeddingGemma)",
                "bert" => "BERT encoder (MiniLM, BGE, E5, GTE, bge-m3)",
                "mistral3" => "Mistral 3 embedding (Nemotron 3.5 Embed)",
                "phi3" => "Phi-3 / Phi-4",
                "qwen3moe" => "Qwen 3 MoE",
                "gpt-oss" => "gpt-oss",
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
    let cpu_str = "x86-64 (AVX2, AVX-VNNI, AVX-512 VNNI/GFNI/VBMI), ARM64 (NEON, SDOT, i8mm); others through scalar code";

    // 4. Tokenizers
    let tok_str = "GPT-2 byte-level BPE (smollm, llama-bpe, qwen2, qwen35, gpt-4o, dbrx, \
         tekken / pixtral, tekken-num3, nemotron-3.5-embed), SentencePiece BPE (llama), \
         WordPiece (bert), SentencePiece unigram (t5)";

    // 5. Chat templates
    let tmpl_str = "read from the model file and rendered exactly as llama.cpp renders them: \
ChatML, Phi-4, Llama 3, Qwen 3, Gemma, Mistral (v0, v1, v7), Llama 2, OpenAI Harmony (gpt-oss) \
— or named with `--chat-template` for a file that declares none";

    // 6. Embeddings
    let embd_str = "causal decoders as embedders (`<arch>.pooling_type`: last or mean), and \
bidirectional encoders: BERT (MiniLM, BGE, E5, GTE, bge-m3: cls/mean), Gemma 3 (EmbeddingGemma: \
mean/cls with `dense_2`/`dense_3`) and Mistral 3 (Nemotron 3.5 Embed 1B: mean); \
`POST /v1/embeddings`, `peregrine embed`";

    // 7. Tool calling
    let tool_str = "OpenAI `tools`, `tool_choice` (`auto`, `none`) and `tool_calls` on \
`POST /v1/chat/completions`, streamed or not, for Qwen 2.5 (the `hermes` tool format); the prompt \
is the one llama.cpp builds from the model's template";

    // 8. Rerank and classify
    let rr_str = "a `bert` file's classifier head (`cls`, `cls.output`, \
`classifier.output_labels`): `POST /v1/rerank` (and `/rerank`, `/v1/reranking`, `/reranking`), \
`POST /v1/classify`, `peregrine rerank`; bge-reranker-v2-m3 and bge-reranker-base checked \
against llama.cpp v0.9.0";

    // 8. Decision models
    let dec_str = "native decision models with a `systemone` template: the `openjev` type, \
`POST /v1/systemone`, `peregrine decide`";

    let _ = writeln!(out, "| | Supported now |");
    let _ = writeln!(out, "|---|---|");
    let _ = writeln!(out, "| Architectures | {arch_str} |");
    let _ = writeln!(out, "| Weight formats | {fmt_str} |");
    let _ = writeln!(out, "| CPUs | {cpu_str} |");
    let _ = writeln!(out, "| Tokenizers | {tok_str} |");
    let _ = writeln!(out, "| Chat templates | {tmpl_str} |");
    let _ = writeln!(out, "| Embeddings | {embd_str} |");
    let _ = writeln!(out, "| Tool calling | {tool_str} |");
    let _ = writeln!(out, "| Rerank and classify | {rr_str} |");
    let _ = writeln!(out, "| Decision models | {dec_str} |");

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
        assert!(text.contains("embeddings:"));
        assert!(text.contains("rerank and classify:"));
        assert!(text.contains("decisions:"));

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
        assert!(md.contains("| Embeddings |"));
        assert!(md.contains("| Rerank and classify |"));
        assert!(md.contains("| Decision models |"));

        #[cfg(feature = "arch-llama")]
        assert!(md.contains("Llama"));
        #[cfg(feature = "fmt-kquant")]
        assert!(md.contains("Q4_K"));
    }
}
