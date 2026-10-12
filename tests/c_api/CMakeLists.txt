include(cc_test)

# Compile production helpers directly: libxllm intentionally hides C++ symbols.
# The Tensor demo separately exercises the exported C ABI.
if(GENERATE_SO)
  cc_test(
    NAME rec_tensor_test
    NO_NPU_INIT
    SRCS
      rec_tensor_test.cpp
      ${PROJECT_SOURCE_DIR}/xllm/c_api/internal/helper.cpp
      ${PROJECT_SOURCE_DIR}/xllm/c_api/internal/rec_tensor.cpp
    DEPS
      :master :util :config :model
      absl::strings Boost::serialization gflags::gflags glog::glog
      Folly::folly nlohmann_json::nlohmann_json GTest::gtest_main
  )
  target_link_libraries(rec_tensor_test PRIVATE
    brpc leveldb::leveldb protobuf::libprotobuf ${OpenCV_LIBS} ${FFMPEG_LIBRARIES})
  add_dependencies(rec_tensor_test brpc-static)
  if(USE_NPU)
    target_link_libraries(rec_tensor_test PRIVATE
      atb_customize unified_dlog hccl c_sec nnopbase ms_tools_ext)
  endif()

  # Exercise the public shared-library entry points without allocating a device.
  cc_test(
    NAME rec_c_api_test
    NO_NPU_INIT
    SRCS rec_c_api_test.cpp
    DEPS xllm GTest::gtest_main
  )
endif()
