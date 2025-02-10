message("### Face Pose CMakeLists.txt ###")
set(CMAKE_CXX_STANDARD 17)

# Import Luxonis Libraries
find_package(depthai 2.25.1 REQUIRED)

add_library(LuxonisCamera luxonis_camera.cpp luxonis_camera.h)
target_link_libraries(LuxonisCamera PUBLIC depthai::opencv ${OpenCV_LIBS})

set(ACTIVE_CAMERA_LIBRARY LuxonisCamera)
add_library(Camera base_camera.h)
target_link_libraries(Camera PUBLIC ${OpenCV_LIBS} ${ACTIVE_CAMERA_LIBRARY})
