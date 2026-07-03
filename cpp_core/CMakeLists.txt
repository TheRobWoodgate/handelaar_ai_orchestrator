cmake_minimum_required(VERSION 3.15...3.28)
project(HandelaarAIOrchestrator VERSION 1.0 LANGUAGES CXX)

# Enforce modern C++ standards for lock-free concurrency and smart pointers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get the EXACT include path for pybind11 (without the -I flag)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(PYBIND11_INCLUDES)

# Define the C++ Core sources (paths relative to this CMakeLists.txt)
set(CORE_SOURCES
    src/lob_engine.cpp
    src/risk_guard.cpp
    src/bindings.cpp
)

# Create the Python module
python3_add_library(cpp_core MODULE ${CORE_SOURCES})

# Include directories (paths relative to this CMakeLists.txt)
target_include_directories(cpp_core PRIVATE include ${PYBIND11_INCLUDE_DIR})

# Aggressive Optimization Flags for HFT Execution
# -O3: Maximum optimization
# -march=native: Optimize for the exact CPU architecture compiling the code (critical for cache alignment)
# -flto: Link-time optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cpp_core PRIVATE -O3 -march=native -ffast-math -Wall -Wextra -Werror)
    set_property(TARGET cpp_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(MSVC)
    target_compile_options(cpp_core PRIVATE /O2 /Ob2 /Oi /Ot /GL /WX)
    set_property(TARGET cpp_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Strip symbols in release mode to reduce binary size and marginally improve load times
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(cpp_core PRIVATE -s)
    endif()
endif()
