# syntax=docker/dockerfile:1.7
#
# A CPU-only image for the OpenAI-compatible server:
#
#   docker build -t peregrine .
#   docker run --rm -p 8201:8201 -v /path/to/models:/models:ro peregrine
#
# The model is read from the /models volume. The default command serves
# /models/model.gguf; pass another path after the image name, or override the
# whole command (see docs/docker.md):
#
#   docker run --rm -p 8201:8201 -v /path/to/models:/models:ro peregrine \
#     -m /models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 8201 --host 0.0.0.0 -t 4

# The builder is pinned to the toolchain rust-toolchain.toml pins, by version
# and by the multi-arch index digest.
FROM rust:1.99.1-slim-bookworm@sha256:f2c004e3d2d8b547e8cf7ffbc29205385f4e71d23e53e4e383065db5e9f822cd AS builder

# Build the facade binary with its default features: every architecture, every
# weight format and the server. (A bare `-p peregrine-server` build has no
# format features and cannot open the common Q4_K_M files.)
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p peregrine \
    && mkdir -p /out \
    && cp /src/target/release/peregrine /out/peregrine

# Runtime: Debian 12's glibc with nothing else, on the image's non-root user
# (uid/gid 65532). Pinned by digest as well.
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f

ARG PEREGRINE_VERSION=dev
LABEL org.opencontainers.image.title="peregrine" \
      org.opencontainers.image.description="CPU inference engine serving a GGUF model behind the OpenAI API" \
      org.opencontainers.image.source="https://github.com/RantAI-dev/peregrine" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${PEREGRINE_VERSION}"

COPY --from=builder --chown=nonroot:nonroot /out/peregrine /usr/local/bin/peregrine

# Models are mounted here. The server binds all interfaces inside the container
# (publishing the port controls who can reach it); the volume is read-only in
# the documented run command.
VOLUME ["/models"]
EXPOSE 8201
USER nonroot:nonroot

ENTRYPOINT ["/usr/local/bin/peregrine", "serve"]
CMD ["-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8201"]
