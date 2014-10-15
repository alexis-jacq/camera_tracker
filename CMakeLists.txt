cmake_minimum_required(VERSION 2.8)

#Déclaration du projet
project(CameraProject)

find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
#Déclaration de l'exécutable
add_executable(camera	src/capture.cpp)
target_link_libraries(camera ${OpenCV_LIBS})


install(TARGETS camera	RUNTIME DESTINATION bin)


