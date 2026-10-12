# REC 0.9 to 0.11 migration

The REC source adaptation is implemented and has passed the validation listed
below. This is a migration candidate, not deployment approval. Historical
predictor AB, batching acceptance and dependency publication remain open.

## Source boundaries

- Source: `4fb40793f2978eae0b7f2f3776872fc2a09ee116` (`release/v0.9.0`).
- Target: `2ed9552a16329a3375bee0ab35425ee546d41e25` (`release/v0.11.0`).
- Branch: `migration/rec-v0.9-to-v0.11`.
- Preserve the 0.11 configuration categories, typed KV cache, grouped input
  metadata, packed worker protocol, scheduler queues, EPLB behavior and general
  ACL graph executor.

This is an adaptation of the REC changes, not an overwrite with the source
branch. The current predictor dependency names a `cedbc8b` package; its source
relationship to the migrated release is unverified.

## Implemented changes

- Direct OneRec Tensor C API, explicit logical device selection and C++ overload
  for the four-argument REC initializer. Inputs are validated and copied into
  request-owned BF16 storage; raw outputs are response-owned. Timeout zero waits
  indefinitely; late responses release their buffers. Completion callbacks are
  guarded against duplicate completion. Timing storage is bounded.
- REC options mapped into 0.11 config categories, including absolute or
  model-relative constrained dictionary paths, graph options and multistream
  settings.
- Cross-attention KV allocation, per-request block tables and cache slots,
  concurrent allocator wrappers and per-engine/per-scheduler-thread input
  scratch space. Worker and scheduler pools join before dependent state is
  destroyed. Completion waits for all rank events.
- Persistent OneRec self-KV/beam buffers and a dedicated OneRec ACL graph
  executor. Both prefill and decode keys include captured cache addresses, including
  after a larger request grows a buffer; each pipeline caches
  at most 32 graphs and uses eager execution for additional shapes.
- CrossAttention ATB operation and Ascend kernel, fused constrained Top-K up to
  512, W8A8 fused/shared-expert loading, and split lm_head loading across shards.
  Split head indices must be contiguous, with offsets inferred from weights.
- Width-one multi-round decoding; the first round creates exactly BW parents.
  Batched first-round output preserves separate request/beam rows.
  Per-token logprobs follow the selected parent history, including repeated
  parents and final output widths different from BW. Invalid parent scores do
  not produce NaN from infinity subtraction.
- Single-step `vocab_probs` output keeps FP32 sampler probabilities after
  filtering, temperature, top-k and top-p. A split first head returns only its
  first segment. Item conversion deduplicates IDs, retains matching metadata,
  and bounds random sampling by the requested item limit.
- A Tensor C API validation demo, raw-output tests, beam/item/timing tests, and
  CrossAttention and constrained Top-K operator tests.
- A shared Python/NPU bootstrap for the server and embedded REC C API. It
  preserves an existing caller's GIL state, serializes initialization and uses
  the idempotent `torch_npu.npu.init()` entry point before device allocation.
- `XLLM_PREBUILT_OPS_DIR` permits an isolated same-source operator library,
  avoiding shared operator installation during validation. ASIO is declared
  for the target Mooncake dependency. TileLang manifests must be generated
  using the target source's `compile-kernels` launcher before CMake configure.

The public C ABI major is **2**. Rebuild predictor using both the headers and
library from this branch. Configuration and dictionary settings remain
process-wide; simultaneous handlers need matching settings.

The migrated CrossAttention kernel requires decoder head dimension 128 and KV
block size 128. Unsupported values fail model initialization. The fused
constrained Top-K path remains opt-in through
`XLLM_ENABLE_ONEREC_FUSED_CONSTRAINED_TOPK` because model comparison found
different ordering for tied scores (see below).

## Validation record — October 12, 2026

- Local formatting of changed C++ ranges and `git diff --check` completed.
- Preliminary C++ syntax checks passed for the migrated C API, REC worker and
  engine, input builders, block layer, graph executors, scheduler, request
  output, sampling, model registration and concurrent block-manager changes.
  These checks reused installed dependency flags; they are not a full build.
- A same-source REC operator binary package and isolated Python extension
  passed all 51 NPU cases (45 constrained Top-K and six CrossAttention), with
  zero skipped cases. CrossAttention uses INT64 actual KV lengths and was
  compared with dense GQA for FP16/BF16, query lengths 1/3/128, discontiguous
  KV pages and unequal encoder lengths.
- Target TileLang generation completed for all ten kernel families. The complete
  48-operator package was built and installed into the isolated vendor directory.
- The production Python/NPU initialization regression passed, including NPU
  allocation, repeated calls with the GIL held, two native callers and normal
  process exit. The full-library CMake version is verified separately.
- Final shared-library and demo CMake builds passed. Twenty-one formal CPU
  cases passed, including three tests linked through the exported C API, beam
  mask/history regressions, and two thread-pool shutdown regressions.
  The final CPU run discovered all 21 cases and passed them without NPU
  allocation, including the batched first-round regression. The formal CMake
  NPU initialization regression passed with normal exit.
- Oxygen loaded its full checkpoint and completed deterministic three-round
  requests. Eager, actual prefill/decode graph capture, and two-worker concurrent
  requests matched exactly, and every returned product code was in the dictionary.
  Integration uncovered and fixed stale CPU beam-prefix masks,
  incorrect REC backend propagation and duplicate static operator registration
  in C API consumers.
- Full-handler teardown exposed an inherited MPMC thread-pool shutdown bug:
  idle semaphore waiters were never awakened before join. The migration wakes
  every waiter and adds idle-exit and queue-drain CPU regressions.
- Width-one dictionary validation caught invalid repeated-token sequences. The
  legacy beam kernel assumes groups of eight beams and K == BW. A device tensor
  fallback preserves parent grouping, sequence history and cache prefix sums for
  unsupported widths/candidate counts. The NPU regression passed 18 combinations:
  BW/K = 1/1, 3/5, 8/8, 16/32, 16/3, 128/128 at steps 1, 2 and 8, each with
  two requests. It verifies selected tokens, cumulative scores, parent indices,
  cache-copy prefix sums and sequence history.
- Oxygen BW=1/K=1, BW=16/K=32, single-step vocabulary probabilities and item
  conversion passed repeatability/shape checks. Product sequences and converted
  IDs were checked against `cupon_700009.bin` (not recommendation quality).
- Encoder length 276 passed eager and graph comparison. Mixed input lengths
  8/276/8 passed serial graph and two-worker concurrent comparison, including
  growth and reuse of graph/cache buffers. Two-worker asynchronous output with
  host-stage serialization disabled also passed all 12 additional requests.
  After removing temporary diagnostics, the final binary passed all 21 CPU
  cases again, six mixed-graph requests and twelve mixed-asynchronous requests
  against the existing serial references.
- Opt-in fused Top-K executed without fallback; its three additional requests
  repeated exactly and all 128 reference products were valid. Comparison with
  composite Top-K reported four step-1 `tie-order` warnings, no `value` warning,
  and maximum compared logprob difference `9.53674e-7`. Token order was not
  bitwise identical. Keep the optimization off by default.

## Batch numerical investigation

A single worker with four callers formed batches of three requests. Both eager
and graph modes differed from serial references on nine of twelve additional
requests. The strict demo check remains nonzero; this is not recorded as a
passing serial-versus-batch test. In one sampled batch, serial/batch product
sets overlapped by 127/128, 125/128 and 124/128: some differences affect product
selection as well as rank.

Layer tracing found identical encoder/context inputs and the first differences
at encoder layer 0. Depending on the input, 28/16384, 155/16384 and
13378/565248 output elements differed there. A separate NPU BF16 linear probe,
using the same first-layer weights and input values without xLLM scheduling,
KV caches or graphs, also changed with full versus split batch projection:
Q/K/V had 117/1/3 differing values, with maximum absolute differences
`0.0009765625`, `0.0009765625` and `0.00006103515625`.

A protected diagnostic then compared the same batch's decoder graph execution
with eager execution. Tokens, decoder context and self-KV were cloned before
the graph call. All nine decoder forwards matched exactly, spanning prefill,
decode and replay; actual graph output sizes were 9 and 384 tokens. Encoder
execution is eager. This supports a BF16 batch-shape sensitivity explanation
for the observed serial/batch differences, but does not establish historical
model equivalence or business acceptance. Temporary tracing/double-forward
code is excluded from the migration commit; the diagnostic patch and results
are retained with the delivery evidence.

Some fused/diagnostic runs emitted TBE subprocess shutdown messages; one
protected diagnostic log also warned about semaphore cleanup. The diagnostic
main process returned the expected serial-golden comparison failure code 1.
These logs do not establish
long-running resource stability; soak testing remains outstanding.

Validation uses a separate directory on 99 in container
`xllm-npu-zhanghe209`, with CANN 9.0.0 and Torch/torch_npu 2.9.0. The original
source/build, globally installed operators and existing services are retained.
Logical `npu:0` with `ASCEND_RT_VISIBLE_DEVICES=4` was verified to map to physical
4; check occupancy again before running tests.

See [rec_tensor_demo.md](rec_tensor_demo.md) for build and invocation examples.
The validation environment reuses an older vcpkg installation without ASIO.
Its Mooncake TCP headers come from yalantinglibs (ASIO 1.24.0), so configure
`ASIO_INCLUDE_DIR=/usr/local/yalantinglibs/include` there as well. Mixing those
headers with an independently selected ASIO 1.29.0 implementation fails linking.
Fresh dependency installations should use the ASIO declared in `vcpkg.json`
consistently for every Mooncake target.
Synthetic embeddings check execution/repeatability only. Real predictor
embeddings and a verified baseline library are needed for a historical AB
comparison. Oxygen graph3 does not supply W8A8 or split-head weight coverage.

## Remaining integration gates

1. Rebuild predictor with the migrated **ABI 2 headers and library together**,
   then compare real captured embeddings/outputs against a verified historical
   package. The current `cedbc8b` package ancestry is not established.
2. Define and validate business acceptance for batch-induced product/rank
   changes. The current implementation does not guarantee batch-invariant
   bitwise results. Keep fused Top-K disabled unless tie-order differences are
   accepted and validated with production inputs.
3. Validate W8A8, split lm_head checkpoints and other model/accelerator paths
   with representative assets; Oxygen's BF16 checkpoint does not cover them.
4. Run sustained concurrency/throughput and teardown/resource stability tests
   before making performance or deployment claims.
5. Publish the pinned dependency commits to accessible forks before ordinary
   recursive cloning/CI from the parent fork. Local bundle restoration is
   provided and independently checked.

## Dependency checkpoints

- `xllm_ops`: `5ecb9620391d0fb12bcd809927ee8a819e4df94b`.
- `xllm_atb_layers`: `e2e85fac8ed0d3255c32943cdbc50f56538771fa`.

Both commits are stored in local Git bundles and have been restored into
independent repositories and compared with the changed source files. They
have not been published. Before normal recursive cloning from the fork can
work, publish these commits to accessible dependency repositories and update
the submodule URLs, or restore the exact commits from the supplied bundles.
