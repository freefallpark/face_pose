message("### Face Pose Estimation CMakeLists.txt ###")
cmake_minimum_required(VERSION 3.20)
project(face_pose_project)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR})

# Compiler Options
# Compiler Options #
add_library(common_flags INTERFACE)
## Set common options
target_compile_options(common_flags INTERFACE -Wall -Wextra -pedantic -Werror)
## Set Debug options
target_compile_options(common_flags INTERFACE $<$<CONFIG:DEBUG>:-fsanitize=address -g -fno-omit-frame-pointer>)
target_link_options(common_flags INTERFACE $<$<CONFIG:DEBUG>:-fsanitize=address>)
## Set Release Options
target_compile_options(common_flags INTERFACE $<$<CONFIG:RELEASE>:-march=native -O2>)

# Google Log
find_package(glog 0.7.1 REQUIRED)
link_libraries(glog::glog)

# Camera
add_subdirectory(camera)
target_link_libraries(Camera PRIVATE common_flags)

# Pose Estimation
add_subdirectory(pose_estimation)
target_link_libraries(FacePose PRIVATE common_flags)

# IPC
add_subdirectory(ipc)
target_link_libraries(FacePoseServer PRIVATE common_flags)
target_link_libraries(FacePoseClient PRIVATE common_flags)

# Process
add_subdirectory(process)
target_link_libraries(FacePoseProcess PRIVATE common_flags)

add_executable(face_pose main.cpp)
target_link_libraries( face_pose  PRIVATE FacePoseProcess)
