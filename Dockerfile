# syntax=docker/dockerfile:1.7

FROM oven/bun:1-alpine AS frontend-builder

WORKDIR /app/admin-ui
COPY admin-ui/package.json admin-ui/bun.lock* ./
RUN --mount=type=cache,id=kiro-rs-bun-cache,target=/root/.bun/install/cache,sharing=locked \
    bun install --frozen-lockfile --ignore-scripts
COPY admin-ui ./
RUN --mount=type=cache,id=kiro-rs-bun-cache,target=/root/.bun/install/cache,sharing=locked \
    bun run build

FROM rust:1.92-alpine AS builder

RUN apk add --no-cache musl-dev perl make

ARG TARGETARCH

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY --from=frontend-builder /app/admin-ui/dist /app/admin-ui/dist

ENV CARGO_HOME=/usr/local/cargo

# Keep Cargo's registry, git checkouts and target directory outside the image
# layer. BuildKit can reuse these caches even when COPY src invalidates the
# cargo build layer (which is the common case during application development).
RUN --mount=type=cache,id=kiro-rs-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=kiro-rs-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=kiro-rs-target-${TARGETARCH},target=/app/target,sharing=locked \
    cargo build --release --locked --no-default-features

FROM alpine:3.21

RUN apk add --no-cache ca-certificates

WORKDIR /app
COPY --from=builder /app/target/release/kiro-rs /app/kiro-rs

VOLUME ["/app/config"]

EXPOSE 8990

CMD ["./kiro-rs", "-c", "/app/config/config.json", "--credentials", "/app/config/credentials.json"]
