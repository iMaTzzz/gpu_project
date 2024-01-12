cmake_minimum_required(VERSION 3.10)
project(gpu_project LANGUAGES C CUDA)

# Set C and CUDA compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -std=c99 -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Iinclude")

# Include directories
include_directories(include)

# Source and CUDA files
file(GLOB SRC_FILES "src/*.c")
file(GLOB CUDA_SRC_FILES "src/*.cu")

# Object files
set(OBJ_FILES "")
foreach(SRC_FILE ${SRC_FILES})
  get_filename_component(OBJ_NAME ${SRC_FILE} NAME_WE)
  list(APPEND OBJ_FILES "obj/${OBJ_NAME}.o")
endforeach()

set(CUDA_OBJ_FILES "")
foreach(CUDA_SRC_FILE ${CUDA_SRC_FILES})
  get_filename_component(CUDA_OBJ_NAME ${CUDA_SRC_FILE} NAME_WE)
  list(APPEND CUDA_OBJ_FILES "obj/${CUDA_OBJ_NAME}.o")
endforeach()

# Prof object files - adjust as needed
set(OBJ_PROF_FILES obj_prof/htables_prof.o obj_prof/bitstream_prof.o)

# Executable
add_executable(ppm2jpeg ${SRC_FILES} ${CUDA_SRC_FILES} ${OBJ_PROF_FILES})
set_target_properties(ppm2jpeg PROPERTIES LINKER_LANGUAGE CXX)

# Link libraries
find_package(CUDA REQUIRED)
target_link_libraries(ppm2jpeg CUDA::m cudart)

# CUDA libraries directories
link_directories(/usr/local/cuda/lib64)

# Custom target for creating object files from C sources
foreach(SRC_FILE ${SRC_FILES})
  get_filename_component(OBJ_NAME ${SRC_FILE} NAME_WE)
  set(OBJ_FILE "obj/${OBJ_NAME}.o")
  add_custom_command(
    OUTPUT ${OBJ_FILE}
    COMMAND ${CMAKE_C_COMPILER} -c ${CMAKE_C_FLAGS} ${SRC_FILE} -o ${OBJ_FILE}
    DEPENDS ${SRC_FILE}
  )
endforeach()

# Custom target for creating object files from CUDA sources
foreach(CUDA_SRC_FILE ${CUDA_SRC_FILES})
  get_filename_component(CUDA_OBJ_NAME ${CUDA_SRC_FILE} NAME_WE)
  set(CUDA_OBJ_FILE "obj/${CUDA_OBJ_NAME}.o")
  add_custom_command(
    OUTPUT ${CUDA_OBJ_FILE}
    COMMAND ${CMAKE_CUDA_COMPILER} -c ${CMAKE_CUDA_FLAGS} ${CUDA_SRC_FILE} -o ${CUDA_OBJ_FILE}
    DEPENDS ${CUDA_SRC_FILE}
  )
endforeach()

# Add clean target
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${OBJ_FILES};${CUDA_OBJ_FILES};ppm2jpeg")

