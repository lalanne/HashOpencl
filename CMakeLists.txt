
cmake_minimum_required (VERSION 2.6) 
project (mxm) 

#add_definitions(-framework OpenCL)
add_definitions(-Wall)

add_executable(hash
            hash.c)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(hash OpenCL)
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")
