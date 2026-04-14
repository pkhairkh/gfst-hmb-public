cmake_minimum_required(VERSION 3.26)

project(gfst_hmb_mpuc_mathdx_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(GFST_HMB_MATHDX_PTXAS_OPT_LEVEL "DEFAULT" CACHE STRING "PTXAS optimization level override for the MathDx extension (DEFAULT, O1, O0)")
set_property(CACHE GFST_HMB_MATHDX_PTXAS_OPT_LEVEL PROPERTY STRINGS DEFAULT O1 O0)
option(GFST_HMB_MATHDX_ENABLE_DEVICE_LTO "Enable device LTO for the MathDx extension" ON)
option(GFST_HMB_MATHDX_ENABLE_IPO "Enable IPO for the MathDx extension" ON)

string(TOUPPER "${GFST_HMB_MATHDX_PTXAS_OPT_LEVEL}" GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED)
if(NOT GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED STREQUAL "DEFAULT" AND
   NOT GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED STREQUAL "O1" AND
   NOT GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED STREQUAL "O0")
  message(FATAL_ERROR "GFST_HMB_MATHDX_PTXAS_OPT_LEVEL must be one of DEFAULT, O1, O0")
endif()

set(_gfst_mathdx_cuda_compile_flags -O3 --expt-relaxed-constexpr --expt-extended-lambda --fmad=false)
if(GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED STREQUAL "O1")
  list(APPEND _gfst_mathdx_cuda_compile_flags -Xptxas=-O1)
elseif(GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED STREQUAL "O0")
  list(APPEND _gfst_mathdx_cuda_compile_flags -Xptxas=-O0)
endif()

if(GFST_HMB_MATHDX_ENABLE_DEVICE_LTO)
  set(_gfst_mathdx_device_lto_value 1)
else()
  set(_gfst_mathdx_device_lto_value 0)
endif()

if(GFST_HMB_MATHDX_ENABLE_IPO)
  set(_gfst_mathdx_ipo_value 1)
else()
  set(_gfst_mathdx_ipo_value 0)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
get_filename_component(TORCH_PACKAGE_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
set(TORCH_PYTHON_LIBRARY "${TORCH_PACKAGE_PREFIX}/lib/libtorch_python.so")

if(NOT DEFINED GFST_HMB_MATHDX_ROOT)
  message(FATAL_ERROR "GFST_HMB_MATHDX_ROOT must be set")
endif()

find_package(mathdx REQUIRED COMPONENTS cublasdx cusolverdx CONFIG PATHS "${GFST_HMB_MATHDX_ROOT}" NO_DEFAULT_PATH)

add_library(gfst_hmb_mpuc_mathdx_ops MODULE
  mathdx_extension.cpp
  canonical_summary_runtime.cu
  mathdx_runtime.cu
)

target_include_directories(gfst_hmb_mpuc_mathdx_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(
  gfst_hmb_mpuc_mathdx_ops
  PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Python
    CUDA::cublas
    CUDA::cusolver
    mathdx::cublasdx
)
if(EXISTS "${TORCH_PYTHON_LIBRARY}")
  target_link_libraries(gfst_hmb_mpuc_mathdx_ops PRIVATE "${TORCH_PYTHON_LIBRARY}")
endif()

if(TARGET mathdx::cusolverdx)
  target_link_libraries(gfst_hmb_mpuc_mathdx_ops PRIVATE mathdx::cusolverdx)
elseif(TARGET mathdx::cusolverdx_fatbin)
  target_link_libraries(gfst_hmb_mpuc_mathdx_ops PRIVATE mathdx::cusolverdx_fatbin)
else()
  message(FATAL_ERROR "No cuSOLVERDx target was found in the MathDx package")
endif()

target_compile_definitions(gfst_hmb_mpuc_mathdx_ops PRIVATE
  TORCH_API_INCLUDE_EXTENSION_H
  GFST_HMB_GLIBCXX_USE_CXX11_ABI=${GFST_HMB_GLIBCXX_USE_CXX11_ABI}
  GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_STR="${GFST_HMB_MATHDX_PTXAS_OPT_LEVEL_NORMALIZED}"
  GFST_HMB_MATHDX_DEVICE_LTO_ENABLED=${_gfst_mathdx_device_lto_value}
  GFST_HMB_MATHDX_IPO_ENABLED=${_gfst_mathdx_ipo_value}
)

target_compile_options(gfst_hmb_mpuc_mathdx_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:${_gfst_mathdx_cuda_compile_flags}>
)
if(GFST_HMB_MATHDX_ENABLE_DEVICE_LTO)
  target_link_options(gfst_hmb_mpuc_mathdx_ops PRIVATE
    $<DEVICE_LINK:-dlto>
  )
endif()

set_target_properties(gfst_hmb_mpuc_mathdx_ops PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  INTERPROCEDURAL_OPTIMIZATION ${GFST_HMB_MATHDX_ENABLE_IPO}
)
