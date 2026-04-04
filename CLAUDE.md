# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sherpa-rs is a Rust binding library for **sherpa-onnx**, a speech processing toolkit built on ONNX Runtime. It provides safe Rust wrappers around C FFI bindings for speech recognition (ASR), text-to-speech (TTS), voice activity detection (VAD), speaker identification, diarization, and audio tagging.

## Build Commands

```bash
cargo build                          # Default build (downloads prebuilt binaries)
cargo build --features "cuda"        # With CUDA support
cargo build --features "directml"    # With DirectML (Windows)
cargo build --features "static"      # Static linking
cargo build --features "build-own"   # Build sherpa-onnx from source via CMake
```

On Linux with static linking: `RUSTFLAGS="-C relocation-model=dynamic-no-pic" cargo build --features "static"`

## Lint

```bash
cargo fmt --check
cargo clippy
```

## Testing

Tests are example-driven (no inline unit tests). CI runs:
```bash
cargo test whisper  # Requires whisper model and test audio in CWD
```

Run a specific example:
```bash
cargo run --example whisper -- <model_path> <audio_path>
```

## Architecture

### Workspace Crates

- **`sherpa-rs-sys`** (`crates/sherpa-rs-sys/`) — Low-level C FFI bindings. Uses `bindgen` to auto-generate Rust bindings from `wrapper.h`. The `build.rs` (~560 lines) handles binary downloading, checksum validation, CMake source builds, and multi-platform linking.
- **`sherpa-rs`** (`crates/sherpa-rs/`) — Safe, ergonomic Rust API. Each audio processing module wraps the sys crate's raw pointers in safe abstractions.

### FFI Wrapper Pattern

Every recognizer module follows the same pattern:
1. **Config struct** — user-friendly Rust configuration
2. **Recognizer struct** — holds a raw C pointer from sherpa-onnx
3. **`new(config)`** — creates the C resource via unsafe FFI
4. **Processing methods** (e.g., `transcribe`) — safe wrappers around unsafe C calls
5. **`Drop` impl** — releases the C resource
6. **`unsafe impl Send + Sync`** — thread safety markers

### Provider Auto-Selection

`get_default_provider()` in `lib.rs` selects the best acceleration backend: CUDA → CoreML (macOS) → DirectML (Windows) → CPU.

### Build System (sherpa-rs-sys)

The `build.rs` has two paths:
- **Default (`download-binaries` feature):** Downloads prebuilt libraries from GitHub releases, validates checksums, caches in platform cache dir. Platform manifest is in `dist.json`.
- **Source build (`build-own` feature):** Builds sherpa-onnx from the git submodule via CMake.

Some feature/platform combos force a source build (e.g., DirectML on Windows, certain static+TTS combos).

Key env vars: `SHERPA_BUILD_DEBUG=1` (verbose build output), `SHERPA_LIB_PATH` (custom library path), `SHERPA_SKIP_GENERATE_BINDINGS` (skip bindgen).

### Feature Flags

- `download-binaries` (default) — use prebuilt libraries
- `tts` (default) — enable TTS modules (vits, kokoro, matcha, kitten, zipvoice)
- `static` — static linking
- `cuda` — NVIDIA CUDA acceleration
- `directml` — DirectML acceleration (Windows)
- `build-own` — build from source
- `sys` — re-export raw C bindings

### Git Submodule

`sherpa-onnx` is vendored as a git submodule at `crates/sherpa-rs-sys/sherpa-onnx`. This provides the C API headers for bindgen and the source for `build-own` builds.
