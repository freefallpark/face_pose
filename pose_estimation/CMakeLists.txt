message("### Face Pose CMakeLists.txt ###")

# Import Luxonis Libraries
find_package(depthai 2.25.1 REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_library(FacePose pose_estimation.cpp)
target_link_libraries(FacePose PUBLIC depthai::opencv ${OpenCV_LIBS})
