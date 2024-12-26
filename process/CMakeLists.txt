message("### Process CMakeLists.txt ###")

add_library(FacePoseProcess process.cpp)
target_link_libraries(FacePoseProcess PRIVATE FacePose FacePoseServer)
