cmake_minimum_required(VERSION 3.24)
project(lmnum LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(LMNUM_FULL_STATIC "Link the executable fully statically (no shared libraries at run time)" ON)
set(LMNUM_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/.deps" CACHE PATH "Directory prepared by scripts/fetch-deps.sh")
set(LMNUM_VERSION "0.1.0")

foreach(dir eigen-3.4.0 boost_1_86_0 pocketfft-4d865cba24359b05b37da4e75aee24498c3b78e7 nanoflann-1.6.1 LBFGSpp-0.3.0)
  if(NOT EXISTS "${LMNUM_DEPS}/src/${dir}")
    message(FATAL_ERROR "${LMNUM_DEPS}/src/${dir} is missing: run scripts/fetch-deps.sh (or set -DLMNUM_DEPS=...)")
  endif()
endforeach()
if(NOT EXISTS "${LMNUM_DEPS}/highs-prefix/lib/libhighs.a")
  message(FATAL_ERROR "HiGHS is not built in ${LMNUM_DEPS}/highs-prefix: run scripts/fetch-deps.sh")
endif()

execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short=12 HEAD OUTPUT_VARIABLE LMNUM_GIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp.tmp"
  "#pragma once\n#define LMNUM_VERSION \"${LMNUM_VERSION}${LMNUM_GIT_SUFFIX}\"\n#define LMNUM_GIT \"${LMNUM_GIT}\"\n")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp.tmp" "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp" COPYONLY)

file(GLOB LMNUM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp")
add_executable(lmnum ${LMNUM_SOURCES})
target_include_directories(lmnum PRIVATE cpp "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_include_directories(lmnum SYSTEM PRIVATE
  "${LMNUM_DEPS}/src/eigen-3.4.0"
  "${LMNUM_DEPS}/src/boost_1_86_0"
  "${LMNUM_DEPS}/src/pocketfft-4d865cba24359b05b37da4e75aee24498c3b78e7"
  "${LMNUM_DEPS}/src/nanoflann-1.6.1/include"
  "${LMNUM_DEPS}/src/LBFGSpp-0.3.0/include"
  "${LMNUM_DEPS}/highs-prefix/include/highs")
# EIGEN_MPL2_ONLY: refuse any Eigen code under a licence other than MPL-2.0. No fast-math: results must be IEEE.
target_compile_definitions(lmnum PRIVATE EIGEN_MPL2_ONLY BOOST_MATH_NO_REAL_CONCEPT_TESTS BOOST_ALL_NO_LIB)
target_compile_options(lmnum PRIVATE -Wall -Wextra -ffp-contract=off)
target_link_libraries(lmnum PRIVATE "${LMNUM_DEPS}/highs-prefix/lib/libhighs.a" pthread m)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE)
  target_link_options(lmnum PRIVATE -static-libstdc++ -static-libgcc)
  if(LMNUM_FULL_STATIC)
    target_link_options(lmnum PRIVATE -static)
  endif()
endif()
