# OneRec Tensor C API validation demo

Build `test_query_rec_input_tensors` with `-DGENERATE_SO=ON`. It uses the
same `xllm_rec_input_tensors_completions` entry point as embedded predictor.
For an isolated NPU build, set `-DXLLM_PREBUILT_OPS_DIR=/absolute/vendor/path`;
when invoking CMake directly, also set
`-DXLLM_ATB_LAYERS_SOURCE_DIR=/absolute/repo/third_party/xllm_atb_layers`
as `setup.py` normally does. The operator directory must contain `op_api/lib/libcust_opapi.so` and the matching
headers, including CrossAttention and RecConstrainedTopK. This option skips
the default shared operator installation. At runtime prepend the same vendor
to `ASCEND_CUSTOM_OPP_PATH` and its `op_api/lib` to `LD_LIBRARY_PATH`.

```sh
cmake --build build --target test_query_rec_input_tensors -j8
build/xllm/test_query_rec_input_tensors --help
```

## Inputs and checks

`--sparse` is a contiguous, native-endian FP32 `[S, hidden]` binary file.
`--context` is an optional FP32 `[C, hidden]` file. The Oxygen graph3 model
uses `hidden=2048` and `C=3`. Use the exact embeddings supplied by predictor
for an end-to-end comparison. Without a sparse file the demo generates small,
deterministic synthetic embeddings; these check execution and repeatability,
not recommendation quality or equivalence to the historical predictor package.

The demo makes one reference request followed by `--requests` additional
requests from `--threads` callers. It checks response status, output shapes,
raw tensor data and exact repeatability. Single-step probability rows are
also checked for finite nonnegative values and a sum of one. `--output` saves
the reference response; `--golden` compares against a previously saved file.
A nonzero exit indicates a request error or a mismatch, which is printed.
When `--output FILE` is supplied, each mismatching response is saved as
`FILE.request-N-variant-V.txt` for token and score comparison.

`--top-k` controls the candidate count (`top_logprobs`) per beam. The demo
keeps sampling `top_k=-1` and `temperature=0` for deterministic beam search.
Use different `--beam-width` and `--top-k` values to exercise the CPU constraint
mask fallback; each decode row must use its surviving beam's current prefix.

`--variant-lengths 8,276,8` creates three distinct synthetic inputs and compares
each request against its own reference in the same handler. It exercises
buffer growth followed by a return to an earlier shape. Combine it with
`--graph`, `--workers` and `--threads` to check graph replay and concurrent
input isolation. With multiple workers the OneRec scheduler issues one request
per worker step. A single worker with several callers can batch requests.
BF16 computation can change with batch size and row placement: Oxygen's
strict serial-versus-batched comparison reports differences. Do not silently
relax this check or use serial output as a bitwise batch-invariant golden.
See the numerical investigation in `rec_migration_status.md`.
It cannot be combined with `--sparse`. Saved multi-input
outputs include the variant index and length; use the same variant order for
`--golden` comparisons.

```sh
# Select an idle physical NPU before launching. Device npu:0 below is logical.
export ASCEND_RT_VISIBLE_DEVICES=4
MODEL=/absolute/path/to/graph3
DEMO=build/xllm/test_query_rec_input_tensors

# Three-round constrained decoding, reference and serial repeatability.
"$DEMO" --model "$MODEL" --device npu:0 \
  --filter cupon_700009.bin --beam-width 128 --top-k 128 --rounds 3 \
  --requests 4 --output /tmp/onerec-eager.txt

# Graph and concurrent execution should preserve the reference result.
"$DEMO" --model "$MODEL" --device npu:0 \
  --filter cupon_700009.bin --beam-width 128 --top-k 128 --rounds 3 \
  --graph 1 --workers 2 --threads 2 --requests 8 \
  --golden /tmp/onerec-eager.txt

# Width-one decoding must still produce all three tokens.
"$DEMO" --model "$MODEL" --filter cupon_700009.bin \
  --beam-width 1 --top-k 1 --rounds 3 --requests 2

# Single-step mode emits vocab_probs; it does not complete a three-token item.
"$DEMO" --model "$MODEL" --filter cupon_700009.bin \
  --beam-width 1 --top-k 128 --rounds 1 --requests 2
```

`--convert-items 1` tests dictionary conversion and aligned item metadata.
Use the same model, dictionary, inputs and sampling configuration for saved
output comparisons. A graph/eager mismatch needs numerical investigation;
this demo deliberately reports it rather than applying an implicit tolerance.
The demo sets block size 128, a 128 MiB KV-cache cap and a 30% device-memory
utilization budget. The generic REC defaults are unsuitable for Oxygen's
paged CrossAttention cache: its 1,000,000-byte cap cannot hold even one block.
The demo does not start a service or alter predictor deployment files.

CPU-only regression targets can be built in the same NPU build. Their
`NO_NPU_INIT` setting prevents the shared test harness from allocating an NPU;
the model demo still exercises initialization through the public C API.

```sh
cmake --build build --target rec_tensor_test rec_c_api_test \
  rec_beam_utils_test rec_item_utils_test rec_infer_timing_test -j8
ctest --test-dir build --output-on-failure \
  -R '^(RecTensorTest|RecOutputTensorTest|RecCApiTest|RecBeamUtilsTest|RecItemUtilsTest|RecInferTimingTest)\.'
```
