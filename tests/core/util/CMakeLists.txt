include(cc_test)

cc_test(
  NAME
    util_test
  SRCS
    blocking_counter_test.cpp
    device_name_utils_test.cpp
    http_downloader_test.cpp
    model_path_utils_test.cpp
    net_test.cpp
    shared_memory_manager_test.cpp
    suffix_decoding_cache_test.cpp
    threadpool_test.cpp
    verbose_trace_logger_test.cpp
    vocab_validation_test.cpp
  DEPS
    util
    :platform
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
)
target_link_libraries(util_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(util_test brpc-static)

cc_test(
  NAME rec_infer_timing_test
  NO_NPU_INIT
  SRCS
    rec_infer_timing_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/util/rec_infer_timing.cpp
  DEPS GTest::gtest_main
)

cc_test(
  NAME mpmc_threadpool_test
  NO_NPU_INIT
  SRCS mpmc_threadpool_test.cpp
  DEPS util GTest::gtest_main
)
