message("### Face Pose Estimation CMakeLists.txt ###")
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

# Pose Estimation
add_subdirectory(pose_estimation)
target_link_libraries(FacePose PRIVATE common_flags)

add_subdirectory(ipc)


add_executable(face_pose main.cpp)
