# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
cmake_minimum_required(VERSION 3.16)
project(model_context_lifetime_regression LANGUAGES CXX)
enable_testing()
find_package(Threads REQUIRED)
get_filename_component(XLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(MODEL_CONTEXT_SOURCE_ROOT "${XLLM_ROOT}" CACHE PATH "Source tree under test")
add_executable(model_context_lifetime_test
  model_context_lifetime_test.cpp
  ${MODEL_CONTEXT_SOURCE_ROOT}/xllm/core/framework/model_context.cpp)
target_compile_features(model_context_lifetime_test PRIVATE cxx_std_17)
target_compile_definitions(model_context_lifetime_test PRIVATE USE_NPU TORCH_HIGHER_THAN_PTA6)
target_include_directories(model_context_lifetime_test PRIVATE
  stubs ${MODEL_CONTEXT_SOURCE_ROOT}/xllm ${MODEL_CONTEXT_SOURCE_ROOT}/xllm/core)
target_link_libraries(model_context_lifetime_test PRIVATE Threads::Threads)
set(MODEL_CONTEXT_TEST_SANITIZER "" CACHE STRING "address,undefined or thread")
if(MODEL_CONTEXT_TEST_SANITIZER)
  target_compile_options(model_context_lifetime_test PRIVATE
    -fsanitize=${MODEL_CONTEXT_TEST_SANITIZER} -fno-omit-frame-pointer)
  target_link_options(model_context_lifetime_test PRIVATE -fsanitize=${MODEL_CONTEXT_TEST_SANITIZER})
endif()
foreach(mode baseline copies derived assignment borrowed empty unwind consumer threads reload)
  add_test(NAME model_context_${mode} COMMAND model_context_lifetime_test ${mode})
  set_tests_properties(model_context_${mode} PROPERTIES TIMEOUT 30)
endforeach()
