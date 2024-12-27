message("### Process CMakeLists.txt ###")

add_library(FacePoseProcess process.cpp)
target_link_libraries(FacePoseProcess PUBLIC FacePose FacePoseServer)
