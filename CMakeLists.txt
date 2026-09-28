# SPDX-License-Identifier: Apache-2.0
# NIRNAY - build system.
#
# Layout of targets:
#   nirnay_objects  object library, compiled once, PIC
#   nirnay_core     static library      (the one the CLI and tests link)
#   nirnay_shared   shared library      (what the Python bindings and the C API consumer use)
#   nirnay-cli      command line front end
#   nirnay_tests    GoogleTest binary
#
# Warning flags are applied PER TARGET, never globally, so that -Werror governs our code
# and not the dependencies fetched below.

cmake_minimum_required(VERSION 3.20)

project(nirnay
        VERSION 0.1.0
        DESCRIPTION "Indigenous GPU-accelerated LP/MILP/QP optimization solver"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------------------

option(NIRNAY_ENABLE_CUDA  "Build the CUDA backend (Phase 4)"          OFF)
option(NIRNAY_BUILD_PYTHON "Build the pybind11 bindings (Phase 10)"    OFF)
option(NIRNAY_BUILD_TESTS  "Build the GoogleTest suite"                ON)
option(NIRNAY_SANITIZE     "Build with AddressSanitizer + UBSan"       OFF)
option(NIRNAY_WITH_ZLIB    "Read gzip-compressed model files"          ON)
option(NIRNAY_WITH_OPENMP  "Multi-core column loops through OpenMP (#57)" ON)

# -----------------------------------------------------------------------------------------
# Build identification. Every benchmark CSV records the commit, so the binary must know it.
# -----------------------------------------------------------------------------------------

find_package(Git QUIET)
set(NIRNAY_GIT_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE NIRNAY_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NIRNAY_GIT_COMMIT STREQUAL "")
    set(NIRNAY_GIT_COMMIT "unknown")
  endif()
endif()

# -----------------------------------------------------------------------------------------
# Dependencies. Per docs/CONTRIBUTING.md: non-solver libraries only. Each entry is justified in
# docs/PROVENANCE.md, which is the artifact a judge reads.
# -----------------------------------------------------------------------------------------

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  GIT_SHALLOW    TRUE)

FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.4.2
  GIT_SHALLOW    TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(fmt cli11 nlohmann_json)

# Dependency headers are pulled into our translation units, so our -Werror set is applied to
# their code as well as ours - fmt 10.2.1's ranges.h trips -Wtautological-compare on GCC 16,
# for instance. Re-export each dependency's include directories as SYSTEM so the compiler
# suppresses diagnostics originating inside them while still policing every line we write.
function(nirnay_mark_include_dirs_system)
  foreach(dep IN LISTS ARGN)
    if(NOT TARGET ${dep})
      continue()
    endif()
    get_target_property(dep_includes ${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
      set_target_properties(${dep} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${dep_includes}")
    endif()
  endforeach()
endfunction()

nirnay_mark_include_dirs_system(fmt CLI11 nlohmann_json)

if(NIRNAY_WITH_ZLIB)
  # Prefer a system zlib; only fetch and build one when the platform has none.
  find_package(ZLIB QUIET)
  if(ZLIB_FOUND)
    message(STATUS "nirnay: using system zlib ${ZLIB_VERSION_STRING}")
    set(NIRNAY_ZLIB_TARGET ZLIB::ZLIB)
  else()
    FetchContent_Declare(zlib
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG        v1.3.1
      GIT_SHALLOW    TRUE)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(zlib)
    # madler/zlib names its static target zlibstatic and does not export a namespaced alias.
    set(NIRNAY_ZLIB_TARGET zlibstatic)
    target_include_directories(zlibstatic PUBLIC
      "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
    message(STATUS "nirnay: building bundled zlib 1.3.1")
  endif()
endif()

if(NIRNAY_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# -----------------------------------------------------------------------------------------
# Warning and sanitizer configuration, applied per target.
# -----------------------------------------------------------------------------------------

add_library(nirnay_flags INTERFACE)

if(MSVC)
  target_compile_options(nirnay_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /Zc:__cplusplus>)
else()
  # C++ only: nvcc does not take these spellings, and the CUDA translation unit gets its own
  # warning set below.
  target_compile_options(nirnay_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wcast-qual
    -Wdouble-promotion
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra --Werror=all-warnings>)
endif()

if(MINGW)
  # Many Windows boxes have an older MinGW earlier on PATH than the toolchain that built
  # the binary; its libstdc++-6.dll is then loaded first and the process dies at start-up
  # with STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139) before main() runs. Linking the runtime
  # statically makes the produced .exe self-contained and removes the whole failure mode.
  target_link_options(nirnay_flags INTERFACE -static-libgcc -static-libstdc++ -static)
endif()

if(NIRNAY_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "NIRNAY_SANITIZE is not supported with MSVC")
  endif()
  target_compile_options(nirnay_flags INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(nirnay_flags INTERFACE -fsanitize=address,undefined)
endif()

# -----------------------------------------------------------------------------------------
# The solver core
# -----------------------------------------------------------------------------------------

set(NIRNAY_CORE_SOURCES
  src/api/c_api.cpp
  src/core/certificate.cpp
  src/core/concurrent.cpp
  src/core/model.cpp
  src/core/solve.cpp
  src/core/solve_control.cpp
  src/io/line_reader.cpp
  src/io/lp_reader.cpp
  src/io/mps_bounds.cpp
  src/io/mps_reader.cpp
  src/io/writer.cpp
  src/qp/convexity.cpp
  src/qp/qp_condat_vu.cpp
  src/la/ldl.cpp
  src/la/lu.cpp
  src/la/scaling.cpp
  src/la/sparse.cpp
  src/simplex/dense_lu.cpp
  src/mip/branch_and_bound.cpp
  src/mip/cuts.cpp
  src/mip/heuristics.cpp
  src/presolve/presolve.cpp
  src/ipm/ipm.cpp
  src/pdhg/cpu_backend.cpp
  src/pdhg/pdhg.cpp
  src/pdhg/residuals.cpp
  src/simplex/crossover.cpp
  src/simplex/dual_simplex.cpp
  src/simplex/primal_simplex.cpp
  src/util/logging.cpp
  src/util/options.cpp
  src/util/version.cpp)

add_library(nirnay_objects OBJECT ${NIRNAY_CORE_SOURCES})

target_include_directories(nirnay_objects PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

# src/ is PRIVATE: it lets one module include another module's implementation header
# (src/mip needs src/simplex, and later phases need more of these) without a chain of
# ../ paths. It is deliberately not PUBLIC - nothing outside the library may depend on
# a header that is not in include/nirnay.
target_include_directories(nirnay_objects PRIVATE "${PROJECT_SOURCE_DIR}/src")

# The concurrent optimizer runs its engines in std::threads (src/core/concurrent.cpp).
find_package(Threads REQUIRED)

target_link_libraries(nirnay_objects
  PUBLIC  fmt::fmt Threads::Threads
  # nlohmann_json is used only by src/io/writer.cpp to emit the benchmark stats blob. It is
  # header-only, so PRIVATE keeps it out of every consumer's include path.
  PRIVATE nlohmann_json::nlohmann_json nirnay_flags)

target_compile_definitions(nirnay_objects PRIVATE
  NIRNAY_VERSION="${PROJECT_VERSION}"
  NIRNAY_GIT_COMMIT="${NIRNAY_GIT_COMMIT}"
  NIRNAY_BUILD_TYPE="$<CONFIG>"
  NIRNAY_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# THE GPU BACKEND (#16-#19). Off by default so a machine with no CUDA toolkit builds the
# whole solver unchanged; with it on, PDHG can run on the device (`--gpu`), and a build on a
# machine with no GPU still runs, warning once and falling back to the CPU. The default
# architectures are Ampere (A100), Hopper (H100) and Blackwell (B200); pass
# -DCMAKE_CUDA_ARCHITECTURES=100 to build for the B200 alone.
if(NIRNAY_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90 100)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  target_sources(nirnay_objects PRIVATE src/gpu/cuda_backend.cu)
  target_link_libraries(nirnay_objects PUBLIC CUDA::cudart CUDA::cusparse)
  target_compile_definitions(nirnay_objects PUBLIC NIRNAY_ENABLE_CUDA)
  message(STATUS "NIRNAY: CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(NIRNAY_WITH_ZLIB AND NIRNAY_ZLIB_TARGET)
  target_link_libraries(nirnay_objects PUBLIC ${NIRNAY_ZLIB_TARGET})
  target_compile_definitions(nirnay_objects PUBLIC NIRNAY_WITH_ZLIB)
endif()

# OPENMP, OPTIONAL AND DETERMINISTIC (#57). Only loops whose every output element is
# written by one thread are parallel - gathers, never reductions - so the answer does not
# depend on the thread count. Without OpenMP the pragmas are ignored and the build is the
# single-threaded one; the `threads` option then reports that it has nothing to set.
if (NIRNAY_WITH_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(nirnay_objects PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(nirnay_objects PUBLIC NIRNAY_HAVE_OPENMP)
    message(STATUS "NIRNAY: OpenMP ${OpenMP_CXX_VERSION} found; column loops run multi-core")
  else()
    message(STATUS "NIRNAY: OpenMP not found; single-threaded build")
  endif()
endif()

add_library(nirnay_core STATIC $<TARGET_OBJECTS:nirnay_objects>)
target_link_libraries(nirnay_core PUBLIC nirnay_objects)

add_library(nirnay_shared SHARED $<TARGET_OBJECTS:nirnay_objects>)
target_link_libraries(nirnay_shared PUBLIC nirnay_objects)
set_target_properties(nirnay_shared PROPERTIES
  OUTPUT_NAME nirnay
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# -----------------------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------------------

add_executable(nirnay-cli apps/nirnay-cli/main.cpp)
target_link_libraries(nirnay-cli
  PRIVATE nirnay_core CLI11::CLI11 nlohmann_json::nlohmann_json nirnay_flags)
set_target_properties(nirnay-cli PROPERTIES OUTPUT_NAME nirnay)

# -----------------------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------------------

if(NIRNAY_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------------------
# Configuration summary - printed so a build log records exactly what was built.
# -----------------------------------------------------------------------------------------

message(STATUS "")
message(STATUS "NIRNAY ${PROJECT_VERSION} (${NIRNAY_GIT_COMMIT})")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA       : ${NIRNAY_ENABLE_CUDA}")
message(STATUS "  sanitizers : ${NIRNAY_SANITIZE}")
message(STATUS "  tests      : ${NIRNAY_BUILD_TESTS}")
message(STATUS "  zlib       : ${NIRNAY_WITH_ZLIB}")
message(STATUS "")
