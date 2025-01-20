message("### Face Pose CMakeLists.txt ###")
set(CMAKE_CXX_STANDARD 17)

# Open CV
find_package(OpenCV 4 REQUIRED)

add_library(FacePose pose_estimation.cpp prior_box.h)
target_link_libraries(FacePose PUBLIC Camera ${OpenCV_LIBS})
