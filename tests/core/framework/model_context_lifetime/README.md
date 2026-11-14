# ModelContext lifecycle regression

Builds the production `model_context.cpp` and header against small ATB, ACL,
Torch and configuration stubs. No NPU or third-party dependency download is
required. The fake runtime checks that each owned context is destroyed once,
on its creation runtime context, and after its last consumer releases it.

```sh
cmake -S tests/core/framework/model_context_lifetime -B /tmp/context-lifetime \
  -DMODEL_CONTEXT_TEST_SANITIZER=address,undefined
cmake --build /tmp/context-lifetime
ctest --test-dir /tmp/context-lifetime --output-on-failure
```

A separate build with `-DMODEL_CONTEXT_TEST_SANITIZER=thread` checks the
cross-thread ownership tests. These tests verify C++ ownership; they do not
substitute for compiling with real CANN headers or running repeated model
updates on NPU. The reload test counts simulated tiling allocations rather
than measuring actual NPU/host allocation sizes.
