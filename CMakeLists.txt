cmake_minimum_required(VERSION 3.16)
project(fem-dev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Gmsh REQUIRED)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_library(fem
    mesh2d.cpp
    ref_T3.cpp
    poisson_kernel.cpp
    assemble.cpp
)

target_include_directories(fem
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fem
    PUBLIC gmsh
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(fem PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
    )
elseif(MSVC)
    target_compile_options(fem PRIVATE /W4)
endif()

enable_testing()

add_executable(test_poisson_solver
    tests/test_poisson_solver.cpp
)


target_link_libraries(test_poisson_solver
    PRIVATE fem
    PRIVATE GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_poisson_solver)