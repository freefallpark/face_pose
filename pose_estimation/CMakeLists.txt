message("### Face Pose CMakeLists.txt ###")
set(CMAKE_CXX_STANDARD 17)

# Import Luxonis Libraries
find_package(depthai 2.25.1 REQUIRED)
find_package(OpenCV 4 REQUIRED)



add_library(FacePose pose_estimation.cpp prior_box.h)
target_link_libraries(FacePose PUBLIC depthai::opencv ${OpenCV_LIBS})

# Copy model into build folder
message("Copying face detection model into build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "Model file name: face_detection_yunet_160x120.blob")
set(MODEL "/home/galileo/reflective_encounters/face_pose/pose_estimation/face_detection_yunet_160x120.blob")
set(CMD "mv ${MODEL} test.blob")
add_custom_command(
        TARGET FacePose POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${MODEL}
                ${CMAKE_BINARY_DIR})
message(STATUS "working directory: ${WORK_DIR}")
message(STATUS "command: ${CMD}")
message(STATUS "result: ${RESULT}")
message(STATUS "error output: ${ERROR_OUTPUT}")

