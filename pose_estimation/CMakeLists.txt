message("### Face Pose CMakeLists.txt ###")
set(CMAKE_CXX_STANDARD 17)

add_library(OpenCVFacePose opencv_face_pose.cpp opencv_face_pose.h)
target_link_libraries(OpenCVFacePose PUBLIC ${OpenCV_LIBS})

set(ACTIVE_POSE_LIBRARY OpenCVFacePose)

add_library(FacePose base_face_pose.h)
target_link_libraries(FacePose PUBLIC ${OpenCV_LIBS} ${ACTIVE_POSE_LIBRARY})
