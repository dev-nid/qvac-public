vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO tetherto/qvac-fabric-llm.cpp
  REF v${VERSION}
  SHA512 d985b9a379c0851a9f3b801dd90d977f0977a64c489daad8c25f0714ba29dec8468d8ea0d67afe7aeffb37ff9bdb62c62d0f49be9cb8a560181bad7376cbb460
)

vcpkg_check_features(
  OUT_FEATURE_OPTIONS FEATURE_OPTIONS
  FEATURES
    force-profiler FORCE_GGML_VK_PERF_LOGGER
    llama BUILD_LLAMA
)

if (VCPKG_TARGET_IS_ANDROID)
  include(${CMAKE_CURRENT_LIST_DIR}/android-vulkan-version.cmake)
  detect_ndk_vulkan_version()
  message(STATUS "Using Vulkan C++ wrappers from version: ${vulkan_version}")
  file(DOWNLOAD
    "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${vulkan_version}.tar.gz"
    "${SOURCE_PATH}/vulkan-sdk-${vulkan_version}.tar.gz"
    TLS_VERIFY ON
  )

  file(ARCHIVE_EXTRACT
    INPUT "${SOURCE_PATH}/vulkan-sdk-${vulkan_version}.tar.gz"
    DESTINATION "${SOURCE_PATH}"
    PATTERNS "*.hpp"
  )

  file(RENAME
    "${SOURCE_PATH}/Vulkan-Headers-${vulkan_version}"
    "${SOURCE_PATH}/ggml/src/ggml-vulkan/vulkan_cpp_wrapper"
  )
endif()

set(PLATFORM_OPTIONS)

if (VCPKG_TARGET_IS_OSX OR VCPKG_TARGET_IS_IOS)
  list(APPEND PLATFORM_OPTIONS -DGGML_METAL=ON)
  if (VCPKG_TARGET_IS_IOS)
    list(APPEND PLATFORM_OPTIONS -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF)
  endif()
elseif(VCPKG_TARGET_IS_ANDROID)
  list(APPEND PLATFORM_OPTIONS -DGGML_VULKAN=ON)
else()
  # Desktop Linux/Windows: CUDA instead of Vulkan
  list(APPEND PLATFORM_OPTIONS -DGGML_VULKAN=OFF -DGGML_CUDA=ON)
  if(VCPKG_TARGET_IS_LINUX)
    # CUDA 12.4 doesn't support libc++ on x86 or Clang 19. Use g++-13 as the
    # CUDA host compiler (highest GCC version supported by CUDA 12.4) and
    # strip -stdlib=libc++ from CUDA flags since GCC doesn't understand it.
    find_program(GXX13 g++-13)
    if(GXX13)
      # g++-13 doesn't understand -stdlib=libc++. Strip it from the vcpkg
      # linker flags so it doesn't leak into CUDA link steps via
      # CMAKE_EXE_LINKER_FLAGS. The C/C++ targets still get libc++ through
      # their own compile/link flags set by the clang toolchain.
      string(REPLACE "-stdlib=libc++" "" VCPKG_LINKER_FLAGS "${VCPKG_LINKER_FLAGS}")
      list(APPEND PLATFORM_OPTIONS
        -DCMAKE_CUDA_HOST_COMPILER=${GXX13}
        -DCMAKE_CUDA_FLAGS=-fPIC)
    endif()
  endif()
endif()

if(VCPKG_TARGET_IS_ANDROID)
  set(DL_BACKENDS ON)
  list(APPEND PLATFORM_OPTIONS
    -DGGML_BACKEND_DL=ON
    -DGGML_CPU_ALL_VARIANTS=ON
    -DGGML_CPU_REPACK=ON)
else()
  set(DL_BACKENDS OFF)
endif()

if (VCPKG_TARGET_IS_ANDROID)
  list(APPEND PLATFORM_OPTIONS
    -DGGML_VULKAN_DISABLE_COOPMAT=ON
    -DGGML_VULKAN_DISABLE_COOPMAT2=ON
    -DGGML_OPENCL=ON)
endif()

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    -DGGML_NATIVE=OFF
    -DGGML_CCACHE=OFF
    -DGGML_OPENMP=OFF
    -DGGML_LLAMAFILE=OFF
    -DLLAMA_MTMD=ON
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_ALL_WARNINGS=OFF
    ${PLATFORM_OPTIONS}
    ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
  PACKAGE_NAME ggml)

if(BUILD_LLAMA)
  vcpkg_cmake_config_fixup(PACKAGE_NAME llama)
endif()

vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()


if(BUILD_LLAMA)
  file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
  file(RENAME "${CURRENT_PACKAGES_DIR}/bin/convert_hf_to_gguf.py" "${CURRENT_PACKAGES_DIR}/tools/${PORT}/convert-hf-to-gguf.py")
  file(INSTALL "${SOURCE_PATH}/gguf-py" DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
  file(RENAME "${CURRENT_PACKAGES_DIR}/bin/vulkan_profiling_analyzer.py" "${CURRENT_PACKAGES_DIR}/tools/${PORT}/vulkan_profiling_analyzer.py")
endif()

if (NOT VCPKG_BUILD_TYPE)
  file(REMOVE "${CURRENT_PACKAGES_DIR}/debug/bin/convert_hf_to_gguf.py")
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (NOT DL_BACKENDS AND VCPKG_LIBRARY_LINKAGE MATCHES "static")
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
