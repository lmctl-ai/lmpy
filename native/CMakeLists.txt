cmake_minimum_required(VERSION 3.24)
# std::to_chars(double) needs macOS 13.3 libc++
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3" CACHE STRING "Minimum macOS version")
project(lmpy LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(LMPY_FULL_STATIC "Also link glibc statically (default: glibc is the only shared dependency)" OFF)
set(LMPY_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/.deps" CACHE PATH "Directory prepared by scripts/fetch-deps.sh")
set(LMPY_VERSION "0.1.0")

foreach(dir eigen-3.4.0 boost_1_86_0 pocketfft-4d865cba24359b05b37da4e75aee24498c3b78e7 nanoflann-1.6.1 LBFGSpp-0.3.0 spectra-1.0.1 xsf-4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b)
  if(NOT EXISTS "${LMPY_DEPS}/src/${dir}")
    message(FATAL_ERROR "${LMPY_DEPS}/src/${dir} is missing: run scripts/fetch-deps.sh (or set -DLMPY_DEPS=...)")
  endif()
endforeach()
if(NOT EXISTS "${LMPY_DEPS}/highs-prefix/lib/libhighs.a")
  message(FATAL_ERROR "HiGHS is not built in ${LMPY_DEPS}/highs-prefix: run scripts/fetch-deps.sh")
endif()

execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short=12 HEAD OUTPUT_VARIABLE LMPY_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp.tmp"
  "#pragma once\n#define LMPY_VERSION \"${LMPY_VERSION}${LMPY_GIT_SUFFIX}\"\n#define LMPY_GIT \"${LMPY_GIT}\"\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp.tmp" "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp" COPYONLY)

file(GLOB LMPY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
add_executable(lmpy ${LMPY_SOURCES})
target_include_directories(lmpy PRIVATE cpp "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(lmpy SYSTEM PRIVATE
  "${LMPY_DEPS}/src/eigen-3.4.0"
  "${LMPY_DEPS}/src/boost_1_86_0"
  "${LMPY_DEPS}/src/pocketfft-4d865cba24359b05b37da4e75aee24498c3b78e7"
  "${LMPY_DEPS}/src/nanoflann-1.6.1/include"
  "${LMPY_DEPS}/src/LBFGSpp-0.3.0/include"
  "${LMPY_DEPS}/src/spectra-1.0.1/include"
  "${LMPY_DEPS}/src/xsf-4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b/include"
  "${LMPY_DEPS}/highs-prefix/include/highs")
# EIGEN_MPL2_ONLY: refuse any Eigen code under a licence other than MPL-2.0. No fast-math: results must be IEEE.
target_compile_definitions(lmpy PRIVATE EIGEN_MPL2_ONLY BOOST_MATH_NO_REAL_CONCEPT_TESTS BOOST_ALL_NO_LIB)
target_compile_options(lmpy PRIVATE -Wall -Wextra -Wno-psabi -ffp-contract=off)
if(WIN32)
  # MinGW: <cmath> constants, a fully static .exe (no libstdc++/winpthread DLLs), and an 8 MiB stack like Linux
  target_compile_definitions(lmpy PRIVATE _USE_MATH_DEFINES)
  target_link_libraries(lmpy PRIVATE "${LMPY_DEPS}/highs-prefix/lib/libhighs.a")
  target_link_options(lmpy PRIVATE -static -Wl,--stack,8388608)
elseif(APPLE)
  target_link_libraries(lmpy PRIVATE "${LMPY_DEPS}/highs-prefix/lib/libhighs.a")
else()
  target_link_libraries(lmpy PRIVATE "${LMPY_DEPS}/highs-prefix/lib/libhighs.a" pthread m)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE AND NOT WIN32)
  target_link_options(lmpy PRIVATE -static-libstdc++ -static-libgcc)
  if(LMPY_FULL_STATIC)
    target_link_options(lmpy PRIVATE -static)
  endif()
endif()
