cmake_minimum_required(VERSION 2.8)

#Déclaration du projet
project(CameraProject)

find_package(OpenCV REQUIRED )

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(facetracker	src/facetracker.cpp)
target_link_libraries(facetracker ${OpenCV_LIBS})


#Déclaration de l'exécutable
add_executable(camera	sample/capture.cpp)
target_link_libraries(camera facetracker ${OpenCV_LIBS})


install(TARGETS camera	RUNTIME DESTINATION bin)


