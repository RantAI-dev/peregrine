# The accuracy sanity check (S5.4d §e)

Not the S5.4c report: this is only "is the body broken". Llama 3.2 1B Q4_K_M, 4 threads,
`--temperature 0`, plain text, no chat template:

```
taskset -c 0-3 target/release/peregrine run -m ~models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "<prompt>" --max-tokens 64 --temperature 0 --threads 4 [--fast-prompt] --json --quiet
```

## Prompt 1

> The lighthouse keeper kept a ledger of every ship that passed the point, noting its name, its
> flag, and the hour.

exact:

> The lighthouse keeper, a man named Thomas, had been keeping a ledger for years, meticulously
> recording every ship that passed the point where the lighthouse stood. He had grown accustomed
> to the routine, and the ledger had become an integral part of his daily routine.
>
> Thomas's ledger was a simple book, bound in worn

fast:

> The lighthouse keeper, a man named Thomas, had been keeping a ledger for years, meticulously
> recording every ship that passed the point where the lighthouse stood. He had grown accustomed
> to the routine, and the ledger had become a trusted companion in his role as a keeper.
>
> Thomas's ledger was a simple book, bound

**First difference:** character 231 (about token 50): exact `"...had become an integral part of
his daily routine."`, fast `"...had become a trusted companion in his role as a keeper."`.

## Prompt 2

> In 1847 a chemist in Uppsala dissolved a mineral in acid and found a new element in the
> residue.

exact:

> That's close to the actual event. In 1847, a chemist in Uppsala, Sweden, named Axel Cronstedt
> discovered a new element. However, he didn't dissolve a mineral in acid to find it.
>
> Axel Cronstedt actually isolated a new element by reacting a mineral with a mixture

fast:

> You're referring to the discovery of Radium.
>
> In 1898, a chemist named Anders Jonas Åkerblad, working in Uppsala, Sweden, discovered a new
> element in the residue of a mineral sample. Åkerblad was experimenting with the mineral
> pitchblende, which is a key

**First difference:** character 0, the first token.

## Why this is not a broken body

The AVX2 body is bit-identical to the scalar twin (the kernel tests, every width class from 1 to
512 activation rows) and the tier's fast goldens — recorded with the scalar twin — pass
unchanged, so the numbers the mode produces here are the numbers the AVX-512 tier produced.

The same comparison between the *exact* engine on this tier and the exact engine under
`PEREGRINE_ISA=scalar` (i.e. no fast mode at all) diverges at character 73 on prompt 1 and at
character 0 on prompt 2 — and on prompt 2 the scalar *exact* run prints the very string the AVX2
*fast* run prints. The engine's own tier-to-tier spread is larger than the fast mode's
approximation on this host, which is the documented per-tier golden effect, not the form.
