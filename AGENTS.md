# AGENTS.md

**Version:** 1.0
**Date:** 2026-09-18
**Purpose:** Technical reference for llama.cpp development

---

## Project Overview

**llama.cpp** is a plain C/C++ implementation of LLM (and VLM) inference with no dependencies, enabling state-of-the-art performance on a wide range of hardware -- locally and in the cloud.

- **Language:** C11 (library, tools, tests) and C++17 (common utilities, server, examples); Python 3.10+ for model conversion scripts and tokenizer tests
- **Architecture:** Library-first modular monolith. The `llama` library (C API in `include/llama.h`, C++ implementation in `src/`) is built on top of the [ggml](https://github.com/ggml-org/ggml) tensor library. Command-line tools and the HTTP server live in `tools/`. Example programs live in `examples/`. Tests live in `tests/`. The `ggml/` subdirectory is a vendored copy of the ggml library with its own CMake build.
- **Purpose:** Run large language and vision models locally on CPU, GPU, and mobile hardware with quantization support (1.5-bit through 8-bit) and backend acceleration (CUDA, Metal, Vulkan, SYCL, etc.)

---

## Quick Setup

```bash
# Build (CPU, default)
cmake -B build
cmake --build build --config Release -j $(nproc)

# Run the CLI
./build/bin/llama-cli -m model.gguf -p "Hello, world!"

# Run C++ tests (CTest, labeled suites)
cd build && ctest -L main --verbose --timeout 900

# Run Python tests
cd build && ctest -L python --verbose

# Install Python dev dependencies (model conversion scripts)
pip install ./gguf-py
```

---

## Architecture

llama.cpp is structured as a layered C/C++ library with optional command-line tools and examples on top.

**Layer 1 -- ggml (vendored in `ggml/`):** The tensor library. Provides backend-agnostic tensor operations and device backends (CPU, CUDA, Metal, Vulkan, etc.). llama.cpp depends on ggml as a subdirectory via `add_subdirectory(ggml)`.

**Layer 2 -- libllama (`src/`):** The core inference library. C API declared in `include/llama.h` and `include/llama-cpp.h`. C++ implementation split across focused modules:

- `llama-model.cpp/h` -- Model loading, architecture dispatch, GGUF parsing
- `llama-context.cpp/h` -- Inference context, evaluation loop, batching
- `llama-sampler.cpp/h` -- Token sampling (temperature, top-k/p, Mirostat, etc.)
- `llama-grammar.cpp/h` -- GBNF grammar-based constrained sampling
- `llama-graph.cpp/h` -- Computation graph construction for model forward pass
- `llama-kv-cache*.cpp/h` -- Key-value cache implementations (ISWA, MSA, recurrent, etc.)
- `llama-hparams.cpp/h` -- Model hyperparameters
- `llama-vocab.cpp/h` -- Tokenizer vocabulary (BPE, SPM, WPM, UGM)
- `llama-quant.cpp/h` -- Quantization types and dequantization
- `unicode.cpp/h` -- Unicode data and text processing

**Layer 3 -- common (`common/`):** Shared utilities used by tools and examples, including:

- `common.cpp/h` -- Argument parsing, model loading wrappers
- `arg.cpp/h` -- Unified argument parser (common to all tools)
- `chat.cpp/h` -- Chat template handling
- `log.cpp/h` -- Structured logging system
- `sampling.cpp/h` -- Sampling logic shared across tools
- `download.cpp/h` -- Model download from Hugging Face
- `json.cpp/h` -- JSON parsing utilities
- `speculative.cpp/h` -- Speculative decoding (draft models)
- `jinja/` -- Jinja template engine for chat templates
- `parsers/` -- PEG parser, auto parser, JSON schema helpers

**Layer 4 -- tools (`tools/`):** Standalone executables, each in its own subdirectory with its own CMakeLists.txt:

- `cli/` -- `llama-cli` (main CLI with chat, embedding, rerank support)
- `completion/` -- `llama-completion` (OpenAI-compatible completions endpoint)
- `server/` -- `llama-server` (HTTP server, REST API, web UI)
- `perplexity/` -- `llama-perplexity` (perplexity evaluation)
- `llama-bench/` -- `llama-bench` (performance benchmarking)
- `quantize/` -- `llama-quantize` (model quantization)
- `mtmd/` -- Multimodal tool
- `tts/` -- Text-to-speech
- And more (tokenize, rpc, export-lora, tuning, fit-params, cvector-generator, etc.)

**Layer 5 -- examples (`examples/`):** Minimal sample programs demonstrating specific use cases (simple, embedding, parallel, speculative, etc.).

**Build system:** CMake is the sole build system (the `Makefile` in the root is a stub that prints an error directing users to CMake). CMake options control backend selection (e.g., `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, `-DLLAMA_BUILD_SERVER=ON`).

---

## Directory Structure

| Path | Purpose |
|------|---------|
| `src/` | Core `llama` library: model loading, context, sampler, grammar, KV cache, vocab, quantization, unicode |
| `include/` | Public C API headers (`llama.h`, `llama-cpp.h`) |
| `ggml/` | Vendored ggml tensor library (own CMakeLists.txt) |
| `common/` | Shared utilities: arg parsing, chat, logging, download, sampling, jinja, parsers |
| `tools/` | Standalone executables: CLI, server, completion, perplexity, bench, quantize, mtmd, tts, etc. |
| `examples/` | Minimal example programs demonstrating API usage |
| `tests/` | C++ test executables registered with CTest (labeled `main`, `python`, `model`) |
| `scripts/` | Python scripts: model conversion, tokenizer utilities, debug helpers |
| `docs/` | Documentation: build guide, model support, completions API, etc. |
| `cmake/` | CMake modules and config files |
| `grammars/` | GBNF grammar definitions for constrained sampling |
| `vendor/` | Third-party vendored code |
| `requirements/` | Python dependency requirement files |
| `gguf-py/` | Python GGUF library used by conversion scripts |
| `.github/workflows/` | CI/CD workflows for multiple backends and platforms |

---

## AI Usage Policy

> [!IMPORTANT]
> AI-generated code is allowed. You are 100% responsible for every line, however it was produced.

Undisclosed AI usage may result in your account being permanently banned from contributing to the project.

If AI is used to generate any portion of the code, contributors must:
1. Explicitly disclose the manner in which AI was employed
2. Check for an existing PR addressing the same change; comment there to avoid duplicates
3. Perform a comprehensive manual review prior to submitting the pull request
4. Be prepared to explain every line of code when asked by a maintainer
5. It is strictly prohibited to use AI to write posts (bug reports, feature requests, PR descriptions, GitHub discussions, responding to humans, etc.)

For full guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Code Style

**C/C++ Conventions:**

- 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Clean trailing whitespaces, end-of-line is LF, end-of-file newline
- 120 character line limit (enforced via clang-format)
- Use sized integer types (`int32_t`, `size_t` for allocation sizes)
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`. In C++ code omit optional `struct` and `enum` keyword when not necessary
- snake_case for function, variable, and type names
- Enum values are always upper case, prefixed with the enum name
- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>` (e.g., `llama_model_init`, `llama_sampler_chain_remove`)
- C/C++ filenames are all lowercase with dashes
- Python filenames are all lowercase with underscores
- Prefer basic `for` loops, avoid fancy STL constructs, avoid templates, keep it simple
- Vertical alignment improves readability and batch editing

**Formatting tools:**

```bash
# Format C/C++ code (clang-format from clang-tools v15+)
clang-format -i src/llama-*.cpp src/llama-*.h

# Check Python code (flake8)
flake8 .

# Type check Python code (mypy, strict mode)
mypy .
```

**File template (C++ header):**
```cpp
#ifndef LLAMA_FOO_H
#define LLAMA_FOO_H

#include "llama-impl.h"

#include <string>
#include <vector>

struct llama_foo {
    // ...
};

#endif // LLAMA_FOO_H
```

**Logging:**

llama.cpp has a structured logging facility in `common/log.h` (`common/log.cpp`). Use it, not bare `fprintf(stderr, ...)`:

```cpp
#include "log.h"

LOG_INF("model", "loaded %d layers\n", n_layers);
LOG_WRN("sampler", "temperature is 0, using greed \n");
LOG_ERR("context", "failed to allocate kv cache\n");
LOG_DER("graph", "op %s: %s = %s\n", op_name, in1, in2); // debug, disabled by default
```

**Log level conventions:**

- **DEBUG** (`LOG_DER`) -- internal state transitions, dispatch details, trim details. Disabled by default, enabled with `--verbose-prompt` or log callbacks
- **INFO** (`LOG_INF`) -- notable events visible in normal operation (model loaded, backend initialized, sampling parameters)
- **WARNING** (`LOG_WRN`) -- something unexpected but non-fatal happened
- **ERROR** (`LOG_ERR`) -- unrecoverable failures where the process cannot continue

---

## Module Naming Conventions

The public API follows a `<class>_<method>` naming pattern with snake_case:

| Prefix | Purpose | Examples |
|--------|---------|----------|
| `llama_model_` | Model loading, architecture, GGUF | `llama_model_load`, `llama_model_quantize` |
| `llama_context_` | Inference context, evaluation | `llama_context_init`, `llama_context_eval` |
| `llama_sampler_` | Token sampling | `llama_sampler_init`, `llama_sampler_sample` |
| `llama_grammar_` | GBNF grammar constraints | `llama_grammar_init`, `llama_grammar_accept` |
| `llama_kv_cache_` | Key-value cache | `llama_kv_cache_init`, `llama_kv_cache_seq_cp` |
| `llama_batch_` | Batch handling | `llama_batch_init`, `llama_batch_add` |
| `ggml_` | Tensor operations | `ggml_mul_mat`, `ggml_new_tensor_1d` |

---

## Testing

llama.cpp uses CMake/CTest for C++ tests and pytest for Python tests. Tests are registered in `tests/CMakeLists.txt` with CTest labels.

**CTest labels:**

- `main` -- Core C++ tests that run without model files
- `python` -- Python-based tests (e.g., `test-jinja-py`)
- `model` -- Tests requiring GGUF model files (run in CI with downloaded models)

**Before Committing:**

```bash
# Build (if not already built)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# Run all main tests (no model files needed)
cd build && ctest -L main --verbose --timeout 900

# Run Python tests
cd build && ctest -L python --verbose

# Run a single test by name
cd build && ctest -R test-tokenizer-0 --verbose

# Run a single test in GDB (debug builds)
gdb --args ./build/bin/test-tokenizer-0 tests/../models/ggml-vocab-llama-spm.gguf

# Lint Python code
flake8 .

# Type check Python code
mypy .

# Format C/C++ code (verify only, no changes)
clang-format --dry-run src/llama-*.cpp src/llama-*.h
```

**Debugging a single test quickly:**

```bash
# Use the debug-test.sh script for fast iteration
./scripts/debug-test.sh test-tokenizer

# Debug in GDB
./scripts/debug-test.sh -g test-tokenizer
```

**Test Locations:**

- `tests/*.cpp` -- C++ test executables (compiled and registered via `llama_build_and_test` / `llama_test` CMake helpers)
- `tests/*.py` -- Python test scripts (tokenizer validation, Jinja template tests)
- `tests/*.sh` -- Shell-based test scripts
- `tests/snapshots/` -- Snapshot data for tests
- `tests/fusion/` -- Operator fusion tests
- `tests/peg-parser/` -- PEG parser tests

**New Feature Checklist:**

1. Add test cases to existing test files or create a new `tests/test-your-feature.cpp`
2. Register the test in `tests/CMakeLists.txt` with appropriate label
3. Build and run `ctest -R test-your-feature` to verify
4. If you modified a ggml operator, add test cases to `test-backend-ops`
5. If you modified model loading, run `test-llama-archs` to verify all architectures

---

## Commit Format

llama.cpp uses squash merges. The squashed commit title follows the format:

```
<module> : <commit title> (#<issue_number>)
```

**Example:**

```bash
git add -A
git commit -m "llama : fix KV being cleared during context shift

Problem: ..."
```

**Module prefixes:** `llama`, `common`, `ggml`, `examples`, `tools`, `docs`, `ci`, `cmake`, etc.
Full module list: https://github.com/ggml-org/llama.cpp/wiki/Modules

---

## Development Tools

**Common Commands:**

```bash
# Build with debug symbols
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_FATAL_WARNINGS=ON
cmake --build build --config Debug -j $(nproc)

# Build with a specific backend (example: CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j $(nprof)

# Build only the library (faster iteration)
cmake -B build -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j $(nproc)

# Generate compile_commands.json for IDE/clangd
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run the full CI test suite locally
LLAMA_FATAL_WARNINGS=ON bash ci/run.sh ./tmp/results ./tmp/mnt

# Debug a specific test
./scripts/debug-test.sh test-tokenizer

# Format C++ code
clang-format -i $(git ls-files '*.cpp' '*.h' '*.c')

# Format Python code
python -m black .  # if black is available

# Install pre-commit hooks
pre-commit install
```

**Backend CMake options:**

| Option | Backend |
|--------|---------|
| `-DGGML_CUDA=ON` | NVIDIA CUDA |
| `-DGGML_METAL=ON` | Apple Metal (macOS/iOS) |
| `-DGGML_VULKAN=ON` | Vulkan |
| `-DGGML_SYCL=ON` | Intel SYCL |
| `-DGGML_HIP=ON` | AMD ROCm |
| `-DGGML_OPENCL=ON` | OpenCL |
| `-DGGML_RPC=ON` | RPC (distributed inference) |
| `-DGGML_BLAS=ON` | BLAS (vendor-specific) |

---

## Common Patterns

**Model loading and inference:**

```cpp
#include "llama.h"

// Initialize backend
llama_backend_init();

// Load model
llama_model * model = llama_model_load("model.gguf", params);
if (!model) { /* handle error */ }

// Create context
llama_context * ctx = llama_init_from_model(model, params);

// Evaluate a batch of tokens
llama_batch batch = llama_batch_init(n_tokens, 0, 1);
llama_batch_add(&batch, token_id, n_past, {0}, true);
llama_decode(ctx, batch);

// Sample next token
llama_token new_token = llama_sampler_sample(ctx, sampler, -1);

// Cleanup
llama_batch_free(batch);
llama_free(ctx);
llama_model_free(model);
llama_backend_free();
```

**Working around the ggml matmul convention:**

```
C = ggml_mul_mat(ctx, A, B)  means  C^T = A B^T  <=>  C = B A^T
```

**Tensor data is stored in row-major order.** Dimension 0 = columns, 1 = rows, 2 = matrices.

---

## Documentation Standards

### Writing User-Facing Documentation

**Tone:** Direct and concise. No corporate fluff.

- **Active voice:** "llama-server starts an HTTP server" not "An HTTP server will be started"
- **Address user directly:** "Set your HF_TOKEN" not "Users should set their HF_TOKEN"
- **Code blocks:** Always specify language for syntax highlighting
- **Examples:** Show both command AND expected output
- Use hyphens (`-`) only, never em-dashes (`--`) or en-dashes (``)

### Keeping Documentation Current

| Change Type | Required Documentation |
|-------------|------------------------|
| New model | Add entry to `docs/models.md` |
| API change | Update `include/llama.h` header comments |
| New tool | Create `tools/<name>/README.md` |
| Build option | Update `docs/build.md` |
| Bug fix | No doc change required unless behavior changed |

**Rule:** Full rewrite, never changelog patches. If a section needs updating, rewrite the entire section.

---

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start, supported backends |
| `AGENTS.md` | Technical reference for agent development |
| `CONTRIBUTING.md` | Contribution guidelines, AI usage policy, coding standards |
| `docs/build.md` | Build instructions for all backends |
| `docs/models.md` | Supported model architectures and formats |
| `docs/completions.md` | Completions API reference |
| `CLAUDE.md` | Pointer to AGENTS.md |
| `tools/*/README.md` | Tool-specific documentation |
| `docs/development/*.md` | Development guides (debugging, parsing, adding models) |

---

## Working Documents

**Purpose:** The `scratch/` directory (gitignored) is the workspace for investigation, analysis, and planning documents.

**Pattern:**

```
Investigation findings -> scratch/ANALYSIS.md (not committed)
Permanent knowledge -> Detailed commit message (committed)
```

Never create working documents in project root -- they clutter the repository.

---

## Anti-Patterns (What NOT To Do)

| Anti-Pattern | Why It's Wrong | What To Do |
|--------------|----------------|------------|
| Skip clang-format on changed files | Inconsistent style, slower review | Run `clang-format -i` on all changed C/C++ files |
| Leave `TODO`/`FIXME` comments in code | Technical debt, incomplete work | Finish implementation before committing |
| Assume code behavior without reading | Causes bugs, breaks things | Read the code, investigate first |
| Create duplicate utility code | Re-implements existing solutions | Search codebase (`grep`, `rg`) for existing implementations |
| Commit without testing | Breaks builds, wastes CI time | Build and run `ctest -L main` before committing |
| Add third-party dependencies | Increases maintenance burden | Reuse existing vendored libraries or ggml primitives |
| Add new data types to `ggml_type` | Disproportionate maintenance burden | Discuss with maintainers first, provide perplexity and performance data |
| Commit large model/snapshot files | Bloats repository | Use git-lfs or external storage, keep snapshots gitignored |
| Use bare `printf`/`fprintf` for logging | Bypasses structured log system | Use `LOG_INF`/`LOG_WRN`/`LOG_ERR` from `common/log.h` |
| Submit features without prior issue discussion | Wastes maintainer time | Open an issue first to gauge interest |
| Write AI-generated PR descriptions | Violates project policy | Let the contributor write their own PR description |

---

## Quick Reference

**C/C++ Build & Test:**
```bash
cmake -B build && cmake --build build -j $(nproc)
cd build && ctest -L main --verbose --timeout 900
```

**Python Lint & Type Check:**
```bash
flake8 .
mypy .
```

**Format Code:**
```bash
clang-format -i $(git ls-files '*.cpp' '*.h' '*.c')
```

**Debug a Single Test:**
```bash
./scripts/debug-test.sh test-tokenizer
```

**Git Operations:**
```bash
git status
git diff
git log --oneline -10
```
