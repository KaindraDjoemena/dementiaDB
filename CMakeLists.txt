cmake_minimum_required(VERSION 3.20)
project(dementiaDB)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED)

set(HIGHFIVE_USE_BOOST   OFF CACHE BOOL "")
set(HIGHFIVE_UNIT_TESTS  OFF CACHE BOOL "")
set(HIGHFIVE_EXAMPLES    OFF CACHE BOOL "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/highfive-devs/HighFive.git
    GIT_TAG v2.9.0
)
FetchContent_MakeAvailable(HighFive)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(dementia_engine STATIC
    src/index.cpp
    src/indexFlat.cpp
    src/indexHnsw.cpp
    src/dementiaDB.cpp
    src/metadata.cpp
)
target_link_libraries(dementia_engine PUBLIC
    HighFive
    ${HDF5_C_LIBRARIES}
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)
target_include_directories(dementia_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${HDF5_C_INCLUDE_DIRS}
)
target_compile_options(dementia_engine PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
)

option(DEMENTIA_BUILD_EXAMPLES "Build examples" ON)
option(DEMENTIA_BUILD_BENCH    "Build benchmarks" ON)

if(DEMENTIA_BUILD_EXAMPLES)
    add_executable(example_save examples/saving.cpp)
    target_link_libraries(example_save PRIVATE dementia_engine)
    target_compile_options(example_save PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)

    add_executable(example_open examples/opening.cpp)
    target_link_libraries(example_open PRIVATE dementia_engine)
    target_compile_options(example_open PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
endif()

if(DEMENTIA_BUILD_BENCH)
    add_executable(hdf5_bench bench/hdf5_bench.cpp)
    target_link_libraries(hdf5_bench PRIVATE dementia_engine)
    target_compile_options(hdf5_bench PRIVATE $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>)
endif()