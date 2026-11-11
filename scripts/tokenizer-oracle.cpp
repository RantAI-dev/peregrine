// Prints llama.cpp's own tokenization and detokenization for a list of test cases,
// so the goldens in crates/peregrine-core/tests/fixtures/ come from the reference
// implementation rather than from Peregrine.
//
// Reads: model path, case file (one escaped case per line).
// Writes per line: bare ids, special ids, escaped text, detokenized text, and the
// ids with special tokens but special parsing off (`parse_special = false`), TAB
// separated. The last column was added for the Tekken fixture; consumers that only
// read the first four columns are unaffected.
//
// Build (from a llama.cpp checkout at cd959ba with build/bin/llama-tokenize):
//   g++ -std=c++17 -O2 -I include -I common -I build/ggml/include \
//       -L build/bin -Wl,-rpath,$PWD/build/bin detok.cpp -o detok -lllama -lggml
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string escape(const std::string & s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02X", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static void print_ids(const std::vector<llama_token> & ids) {
    printf("[");
    for (size_t i = 0; i < ids.size(); ++i) {
        printf(i ? ", %d" : "%d", ids[i]);
    }
    printf("]");
}

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s MODEL.gguf CASES.txt [KEY=VALUE]\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    std::vector<llama_model_kv_override> overrides(2); // the loader walks to a zero key
    if (argc == 4) {
        // A string override of an existing metadata key, e.g.
        // `tokenizer.ggml.pre=nemotron-3.5-embed`. `--override-kv` exists in the CLI
        // but the tokenize tool does not pass it on to the loader.
        const char * eq = strchr(argv[3], '=');
        if (!eq || (size_t) (eq - argv[3]) >= sizeof(overrides[0].key)) {
            fprintf(stderr, "bad override: %s\n", argv[3]);
            return 2;
        }
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        memcpy(overrides[0].key, argv[3], (size_t) (eq - argv[3]));
        snprintf(overrides[0].val_str, sizeof(overrides[0].val_str), "%s", eq + 1);
        mp.kv_overrides = overrides.data();
    }
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    std::vector<std::string> cases;
    std::string cur;
    for (int ch; (ch = fgetc(f)) != EOF;) {
        if (ch == '\n') {
            cases.push_back(cur);
            cur.clear();
        } else {
            cur += static_cast<char>(ch);
        }
    }
    if (!cur.empty()) {
        cases.push_back(cur);
    }
    fclose(f);

    for (auto & escaped : cases) {
        std::string text = escaped;
        // Same escapes llama.cpp's common_params_parse applies to -p.
        std::string unescaped;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\\' && i + 1 < text.size()) {
                char c = text[++i];
                switch (c) {
                    case 'n': unescaped += '\n'; break;
                    case 'r': unescaped += '\r'; break;
                    case 't': unescaped += '\t'; break;
                    case '\\': unescaped += '\\'; break;
                    case 'x':
                        if (i + 2 < text.size()) {
                            char hex[3] = { text[i + 1], text[i + 2], 0 };
                            char * end = nullptr;
                            long v = strtol(hex, &end, 16);
                            if (end == hex + 2) {
                                i += 2;
                                unescaped += static_cast<char>(v);
                                break;
                            }
                        }
                        unescaped += "\\x";
                        break;
                    default: unescaped += '\\'; unescaped += c; break;
                }
            } else {
                unescaped += text[i];
            }
        }

        std::vector<llama_token> bare(llama_vocab_n_tokens(vocab) + 16);
        int n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), bare.data(),
                               (int) bare.size(), /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) {
            bare.resize(-n);
            n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), bare.data(),
                               (int) bare.size(), false, true);
        }
        if (n < 0) {
            fprintf(stderr, "tokenize failed for %s\n", escaped.c_str());
            return 1;
        }
        bare.resize(n);

        std::vector<llama_token> special(llama_vocab_n_tokens(vocab) + 16);
        n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), special.data(),
                           (int) special.size(), /*add_special*/ true, /*parse_special*/ true);
        if (n < 0) {
            special.resize(-n);
            n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), special.data(),
                               (int) special.size(), true, true);
        }
        if (n < 0) {
            fprintf(stderr, "tokenize(special) failed for %s\n", escaped.c_str());
            return 1;
        }
        special.resize(n);

        std::vector<llama_token> noparse(llama_vocab_n_tokens(vocab) + 16);
        n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), noparse.data(),
                           (int) noparse.size(), /*add_special*/ true, /*parse_special*/ false);
        if (n < 0) {
            noparse.resize(-n);
            n = llama_tokenize(vocab, unescaped.data(), (int) unescaped.size(), noparse.data(),
                               (int) noparse.size(), true, false);
        }
        if (n < 0) {
            fprintf(stderr, "tokenize(no parse) failed for %s\n", escaped.c_str());
            return 1;
        }
        noparse.resize(n);

        std::string detok(bare.size() * 8 + 16, '\0');
        int m = llama_detokenize(vocab, bare.data(), (int) bare.size(), &detok[0],
                                 (int) detok.size(), /*remove_special*/ false,
                                 /*unparse_special*/ true);
        if (m < 0) {
            detok.resize(-m);
            llama_detokenize(vocab, bare.data(), (int) bare.size(), &detok[0], (int) detok.size(),
                             false, true);
            m = -m;
        }
        detok.resize(m);

        print_ids(bare);
        printf("\t");
        print_ids(special);
        printf("\t%s\t%s\t", escaped.c_str(), escape(detok).c_str());
        print_ids(noparse);
        printf("\n");
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
