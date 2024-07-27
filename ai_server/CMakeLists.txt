cmake_minimum_required(VERSION 3.10)
execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE ARCHITECTURE
  OUTPUT_STRIP_TRAILING_WHITESPACE)


  # 项目名称
  project(ice_project)
  set(AIServer_Name "ai_server")
  # 启动对C++11标准的支持
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_BUILD_TYPE "Debug")
  # set(CMAKE_BUILD_TYPE "Release")
  # set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/x86)

  set(OpenCV_DIR /usr/local/opencv4.9.0)
  find_package(OpenCV 4.9.0 REQUIRED)

  add_compile_options(-std=c++11 -lpthread)
  include_directories(${OpenCV_INCLUDE_DIRS})
  include_directories(include)


  link_directories( /usr/local/opencv4.9.0/lib)

  add_executable(
    ${AIServer_Name}
    server.cpp
    src/objDetect.cpp
    src/systemCfgTool.cpp
    src/commonTools.cpp
    )
    target_link_libraries(
      ${AIServer_Name}
      ${OpenCV_LIBS})