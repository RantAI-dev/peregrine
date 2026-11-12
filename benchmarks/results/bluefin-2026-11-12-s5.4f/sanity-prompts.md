# The two plain-text prompts of the sanity check (S5.4f §e)

The same two S5.4d used, no chat template, 64 greedy tokens, Llama 3.2 1B Q4_K_M, 4 threads:

```
taskset -c 0-3 target/release/peregrine run -m ~models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "<prompt>" --max-tokens 64 --temperature 0 --threads 4 [--fast-prompt] --json --quiet
```

## Prompt 1

> The lighthouse keeper kept a ledger of every ship that passed the point, noting its name, its
> flag, and the hour.

## Prompt 2

> In 1847 a chemist in Uppsala dissolved a mineral in acid and found a new element in the
> residue.

The four outputs (exact and fast for each) are in `sanity.txt`.
