cmake_minimum_required(VERSION 2.6)
project(render_server)

find_package(OpenGL REQUIRED)

set(PUBLIC_DOCS
    README.md
)

set(SOURCES
  src/main.cc
  src/utils.cc
  src/renderer.cc
  src/camera.cc
  src/scene.cc
  src/object.cc
  src/shader.cc
  src/light.cc
  external/json/jsoncpp.cpp
  external/glad/glad.c
  external/tiny_obj_loader/tiny_obj_loader.cc
)

include_directories(
    external/
    external/tiny_obj_loader/
)

set(LIBRARIES
    glfw
    X11
    pthread
    dl
    m
)

add_compile_options(-std=c++11)

add_executable(${PROJECT_NAME}
    ${SOURCES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES} -Xlinker --unresolved-symbols=ignore-in-shared-libs)

