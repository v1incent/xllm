/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/npu/python_runtime.h"

#include <Python.h>
#include <acl/acl.h>
#include <glog/logging.h>

#include <cstdlib>
#include <mutex>
#include <string>

namespace xllm::npu {
namespace {

// Release a Python caller's GIL before waiting for the initialization mutex.
// Otherwise another initializer holding that mutex could wait for our GIL.
class CallerGilRelease final {
 public:
  CallerGilRelease() {
    if (Py_IsInitialized() && PyGILState_Check()) {
      state_ = PyEval_SaveThread();
    }
  }
  ~CallerGilRelease() {
    if (state_ != nullptr) {
      PyEval_RestoreThread(state_);
    }
  }
  CallerGilRelease(const CallerGilRelease&) = delete;
  CallerGilRelease& operator=(const CallerGilRelease&) = delete;

 private:
  PyThreadState* state_ = nullptr;
};

void acquire_gil_at_exit() {
  // torch_npu owns Python objects in static storage. Keep the GIL held while
  // those objects are destroyed at process exit, without finalizing Python.
  if (Py_IsInitialized()) {
    static_cast<void>(PyGILState_Ensure());
  }
}

}  // namespace

bool init_python_runtime(int32_t device_index) {
  if (device_index < 0) {
    return false;
  }
  // Match the server's torch_npu bootstrap, but execute in a private namespace
  // so an embedded caller's __main__ globals are preserved.
  const std::string script =
      "import os\n"
      "os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'\n"
      "import torch\n"
      "orig = torch._C._get_accelerator\n"
      "try:\n"
      "    torch._C._get_accelerator = lambda: torch.device('cpu')\n"
      "    import torch_npu\n"
      "finally:\n"
      "    torch._C._get_accelerator = orig\n"
      "import torch_npu.npu as _npu_mod\n"
      // The public initializer guards repeated calls and drains queued setup.
      // Calling _C._npu_init repeatedly corrupts Torch Python type ownership.
      "_npu_mod.init()\n"
      "torch_npu._C._npu_setDevice(" +
      std::to_string(device_index) + ")\n";

  CallerGilRelease caller_gil_release;
  static std::mutex runtime_mutex;
  std::lock_guard<std::mutex> lock(runtime_mutex);
  const aclError acl_ret = aclInit(nullptr);
  if (acl_ret != ACL_SUCCESS && acl_ret != ACL_ERROR_REPEAT_INITIALIZE) {
    LOG(ERROR) << "aclInit failed with error " << acl_ret;
    return false;
  }
  const bool initialized_here = !Py_IsInitialized();
  if (initialized_here) {
    Py_InitializeEx(/*initsigs=*/0);
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* globals = PyDict_New();
  PyObject* result = nullptr;
  if (globals != nullptr &&
      PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) ==
          0) {
    result = PyRun_String(script.c_str(), Py_file_input, globals, globals);
  }
  const bool success = result != nullptr;
  if (!success) {
    PyErr_Print();
    LOG(ERROR) << "torch_npu Python runtime initialization failed";
  }
  Py_XDECREF(result);
  Py_XDECREF(globals);
  if (initialized_here) {
    CHECK_EQ(std::atexit(acquire_gil_at_exit), 0);
  }
  PyGILState_Release(gil);
  if (initialized_here) {
    PyEval_SaveThread();
  }
  return success;
}

}  // namespace xllm::npu
