# Security policy

Peregrine parses model files and HTTP requests from sources you may not control. Bugs in either can be
security bugs: memory-unsafety in the GGUF reader or a kernel, a crafted file that reads out of bounds, or a
request that exhausts memory or crashes the server.

## Reporting a vulnerability

Please report privately through GitHub: **Security → Report a vulnerability** on
https://github.com/RantAI-dev/peregrine (private vulnerability reporting). Do not open a public issue.

Include what you found, how to reproduce it (a small model file or request if possible), and the impact you
expect. We aim to acknowledge within 3 working days and to agree on a fix and disclosure date with you.

## Supported versions

Until 1.0, only the latest release and `main` receive security fixes.

## Scope notes

- Model files are treated as untrusted input: a malformed GGUF must produce an error, never undefined
  behaviour. Reports of crashes on crafted files are in scope.
- `peregrine-server` binds to `127.0.0.1` by default and has no authentication. Exposing it on a network
  without a proxy that authenticates is a deployment choice, not a vulnerability.
