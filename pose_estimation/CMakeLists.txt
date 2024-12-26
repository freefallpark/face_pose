message("### Face Pose CMakeLists.txt ###")

# Import Luxonis Libraries
find_package(depthai 2.25.1 REQUIRED)

add_library(FacePose pose_estimation.cpp)
target_link_libraries(FacePose PRIVATE depthai::opencv)
