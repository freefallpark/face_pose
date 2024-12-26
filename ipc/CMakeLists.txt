message("### IPC CMakeLists.txt ###")

# DDS Middleware #
set(IDL_PATH "${CMAKE_SOURCE_DIR}/ipc/idl")
## Run IDL AutoGen
execute_process(WORKING_DIRECTORY ${IDL_PATH}
        COMMAND bash ${IDL_PATH}/generate_idl.sh ${IDL_PATH}
        RESULT_VARIABLE result
        ERROR_VARIABLE error_output)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "FastDDS Autogen Failed, make sure generate_idl.sh has executable permissions (chmod +x
    <path_to_generate_idl.sh>:
${error_output}")
endif()
set(AUTOGEN_PATH ${IDL_PATH}/autogen)
file(GLOB AUTOGEN_SRCS "${AUTOGEN_PATH}/pose*")
message(STATUS "Autogen Pose Sources: " ${AUTOGEN_SRCS})

## Fast-DDS Libraries ##
find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)

# Client Library
set(POSE_CLIENT_SRCS client.cpp client.h ${AUTOGEN_SRCS})
add_library(FacePoseClient ${POSE_CLIENT_SRCS})

# Server Library
set(POSE_SERVER_SRCS server.cpp server.h ${AUTOGEN_SRCS})
add_library(FacePoseServer ${POSE_SERVER_SRCS})
