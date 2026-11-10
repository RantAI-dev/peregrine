// Prints the token ids llama.cpp's rerank path builds for a list of (query, document)
// pairs, using llama.cpp's own `format_prompt_rerank` from tools/server/server-common.cpp
// rather than a re-implementation.
//
// Reads: model path, pairs file (query<TAB>document, one escaped pair per line).
// Writes per line: the ids, as a Python-style list.
#include "server-common.h"

#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string unescape(const std::string & text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            char c = text[++i];
            switch (c) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case '\\': out += '\\'; break;
                case 'x':
                    if (i + 2 < text.size()) {
                        char hex[3] = { text[i + 1], text[i + 2], 0 };
                        char * end = nullptr;
                        long v = strtol(hex, &end, 16);
                        if (end == hex + 2) {
                            i += 2;
                            out += static_cast<char>(v);
                            break;
                        }
                    }
                    out += "\\x";
                    break;
                default: out += '\\'; out += c; break;
            }
        } else {
            out += text[i];
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf PAIRS.txt\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * f = fopen(argv[2], "rb");
    if (!f) {
        return 1;
    }
    std::string all;
    for (int ch; (ch = fgetc(f)) != EOF;) {
        all += static_cast<char>(ch);
    }
    fclose(f);

    size_t pos = 0;
    while (pos <= all.size()) {
        size_t nl = all.find('\n', pos);
        if (nl == std::string::npos) {
            nl = all.size();
        }
        std::string line = all.substr(pos, nl - pos);
        pos = nl + 1;
        if (line.empty() && nl == all.size()) {
            break;
        }
        size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            fprintf(stderr, "malformed pair line\n");
            return 1;
        }
        const std::string query = unescape(line.substr(0, tab));
        const std::string doc = unescape(line.substr(tab + 1));

        mtmd_helper_init_opt init_opt{};
        server_tokens tokens =
            format_prompt_rerank(model, vocab, nullptr, query, doc, {}, {}, init_opt);
        const llama_tokens & ids = tokens.get_tokens();
        printf("[");
        for (size_t i = 0; i < ids.size(); ++i) {
            printf(i ? ", %d" : "%d", ids[i]);
        }
        printf("]\n");
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
