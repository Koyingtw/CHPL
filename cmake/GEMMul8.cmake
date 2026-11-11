# Builds a minimal static GEMMul8 library (only the gemm<double, INT8> path
# that CHPL uses: 9 object files) from the gemmul8/ git submodule.
#
# A full `make` of GEMMul8 compiles hundreds of template instantiations and
# can exhaust compile resources, so the instantiation lists are restricted
# here.
#
# Provides the imported target CHPL::gemmul8.

set(GEMMUL8_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/gemmul8" CACHE PATH
    "Path to the GEMMul8 source tree")
set(GEMMUL8_JOBS 4 CACHE STRING "Parallel jobs used to build GEMMul8")

if(NOT EXISTS "${GEMMUL8_ROOT}/Makefile")
  message(FATAL_ERROR
    "GEMMul8 not found at ${GEMMUL8_ROOT}.\n"
    "Run: git submodule update --init --recursive")
endif()

# CUDA toolkit root as seen by GEMMul8's Makefile (it uses $(CUDA_PATH)/bin/nvcc)
get_filename_component(_chpl_cuda_path "${CUDAToolkit_BIN_DIR}" DIRECTORY)

# GPU_ARCH for GEMMul8: first numeric part of CMAKE_CUDA_ARCHITECTURES (90a -> 90)
list(GET CMAKE_CUDA_ARCHITECTURES 0 _chpl_arch0)
string(REGEX MATCH "^[0-9]+" GEMMUL8_GPU_ARCH "${_chpl_arch0}")

set(GEMMUL8_LIBRARY "${GEMMUL8_ROOT}/lib/libgemmul8.a")

set(_chpl_g8_inst
  "src/gemm/cu_recipe/instantiations.mk \
src/oz2/scaling/accu/cu_recipe/instantiations.mk \
src/oz2/scaling/fast/cu_recipe/instantiations.mk \
src/oz2/scaling/general/cu_recipe/instantiations.mk \
src/oz2/mod/cu_recipe/instantiations.mk \
src/oz2/undo_scaling/cu_recipe/instantiations.mk \
src/worksize/cu_recipe/instantiations.mk \
src/trsm/cu_recipe/instantiations.mk")

add_custom_command(
  OUTPUT "${GEMMUL8_LIBRARY}"
  COMMAND make -C "${GEMMUL8_ROOT}" lib/libgemmul8.a -j${GEMMUL8_JOBS}
    BACKEND=cuda "CUDA_PATH=${_chpl_cuda_path}" GPU_ARCH=${GEMMUL8_GPU_ARCH}
    "INST_MK=${_chpl_g8_inst}"
    GEMM_TYPE_TRIPLES_REAL=d_d_d GEMM_TYPE_TRIPLES_COMPLEX= GEMMLT_BACKEND_CODES=
    ACCU_TYPE_CODES_REAL=d ACCU_TYPE_CODES_COMPLEX= ACCU_BACKEND_CODES=i8
    ACCU_TRI_PATTERNS=full_nonunit ACCU_UPLO_CODES= ACCU_OUT_UPLO_CODES=out_full
    ACCU_OUT_UPLO_CODES_TRI=
    FAST_TYPE_CODES_REAL=d FAST_TYPE_CODES_COMPLEX= FAST_BACKEND_CODES=i8
    FAST_TRI_PATTERNS=full_nonunit FAST_UPLO_CODES=
    GENERAL_TYPE_CODES_REAL=d GENERAL_TYPE_CODES_COMPLEX= GENERAL_BACKEND_CODES=i8
    GENERAL_TRI_PATTERNS=full_nonunit GENERAL_UPLO_CODES=
    TRSM_TYPE_PAIRS_REAL= TRSM_TYPE_PAIRS_COMPLEX= TRSM_LT_BACKEND_CODES=
    MOD_COMPLEX_CODES=real MOD_BACKEND_CODES=i8 MOD_UPLO_CODES= MOD_UPLO_CODES_WITH_FULL=full
    UNDO_TRIPLES=d_d_d UNDO_BACKEND_CODES=i8 UNDO_UPLO_CODES= UNDO_UPLO_CODES_WITH_FULL=full
    UNDO_SYR2K_TRIPLES= UNDO_HER2K_TRIPLES=
    WORKSIZE_COMPLEX_CODES=real WORKSIZE_BACKEND_CODES=i8
  COMMENT "Building minimal GEMMul8 (gemm<double, INT8>)"
  VERBATIM)
add_custom_target(gemmul8_build DEPENDS "${GEMMUL8_LIBRARY}")

add_library(CHPL::gemmul8 STATIC IMPORTED GLOBAL)
set_target_properties(CHPL::gemmul8 PROPERTIES
  IMPORTED_LOCATION "${GEMMUL8_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${GEMMUL8_ROOT}/include")
add_dependencies(CHPL::gemmul8 gemmul8_build)
