cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0072 NEW)
project(awakening LANGUAGES CXX C)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_BUILD_TYPE "Release")
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

#修改以下option需要rebuild清除缓存才能生效
option(BUILD_WITH_TRT      "Enable TensorRT backend"   OFF)
option(BUILD_WITH_OPENVINO "Enable OpenVINO backend"   ON)
option(BUILD_SENTRY "Enable Sentry"   ON)
option(BUILD_RADAR "Enable Radar"   OFF)
option(BUILD_RUNTIME_TESTS "Build runtime test executables by default" ON)
option(BUILD_WITH_PCH "Enable precompiled headers for C++ targets" ON)
option(BUILD_WITH_Daedalus "Enable Daedalus interface" ON)
option(BUILD_WITH_RERUN "Enable Rerun visualization when the SDK is available" ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(AWAKENING_PCH_HEADER ${PROJECT_SOURCE_DIR}/src/pch.hpp)

function(awakening_enable_pch target_name)
    if(BUILD_WITH_PCH AND COMMAND target_precompile_headers)
        target_precompile_headers(
            ${target_name}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${AWAKENING_PCH_HEADER}>"
        )
    endif()
endfunction()


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(HikSDK QUIET)
find_package(MvSDK QUIET)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/hy/Galaxy_Linux-x86_Gige-U3_32bits-64bits_2.6.2606.9251/Galaxy_camera")
find_package(DahengSDK QUIET)
set(BUILD_DEFINITIONS "")
set(BUILD_LIBS "")
if(HikSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_HikSDK)
    list(APPEND BUILD_LIBS HikSDK::HikSDK)
endif()

if(MvSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_MvSDK)
    list(APPEND BUILD_LIBS MvSDK::MvSDK)
endif()

if(DahengSDK_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_DahengSDK)
    list(APPEND BUILD_LIBS DahengSDK::DahengSDK)
endif()

set(RERUN_FOUND FALSE)
if(BUILD_WITH_RERUN)
    find_package(rerun_sdk CONFIG QUIET)
    if(TARGET rerun_sdk)
        set(RERUN_FOUND TRUE)
        message(STATUS "Rerun SDK found, enabling the Rerun visualization backend")
    else()
        message(WARNING "Rerun SDK not found; Web/ROS2 visualization remains available")
    endif()
endif()



if(RERUN_FOUND)
    list(APPEND BUILD_DEFINITIONS USE_RERUN)
endif()
if(BUILD_WITH_Daedalus)
    list(APPEND BUILD_DEFINITIONS USE_Daedalus)
endif()

if(BUILD_WITH_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
    list(APPEND BUILD_DEFINITIONS USE_OPENVINO)
    list(APPEND BUILD_LIBS openvino::runtime openvino::frontend::onnx)
endif()

if(BUILD_WITH_TRT)
    # TensorRT is installed outside this repository.  Prefer an explicit
    # TENSORRT_ROOT so the same checkout works on different machines instead of
    # hard-coding another user's /home/<name>/TensorRT-* path.
    if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{TENSORRT_ROOT}")
        set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT SDK root")
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)
    include_directories(${TensorRT_INCLUDE_DIR} ${TensorRT_INCLUDE_DIRS})
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
    endif()
    # RTX 50-series is sm_120; native lets the installed CUDA toolkit select it.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    list(APPEND BUILD_DEFINITIONS USE_TRT)
    if(TARGET CUDA::nppicc)
        list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart CUDA::nppicc)
    else()
        list(APPEND BUILD_LIBS TensorRT::TensorRT TensorRT::nvonnxparser CUDA::cudart "${CUDAToolkit_LIBRARY_DIR}/libnppicc.so")
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/param_deliver.h.in  
    ${CMAKE_BINARY_DIR}/src/param_deliver.h            
    @ONLY
)
add_library(relink SHARED src/relink.cpp)
awakening_enable_pch(relink)
target_link_libraries(relink Eigen3::Eigen)
include_directories(${PROJECT_NAME}_utils PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${CMAKE_BINARY_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
file(GLOB_RECURSE UTILS_SOURCES  src/utils/*.cpp 3rdparty/*.cpp 3rdparty/*.c)
list(FILTER UTILS_SOURCES EXCLUDE REGEX "/3rdparty/backward-cpp/(test|test_package)/")
if(BUILD_WITH_TRT)
    file(GLOB_RECURSE UTILS_CUDA_SOURCES  src/utils/*.cu)
    list(APPEND UTILS_SOURCES ${UTILS_CUDA_SOURCES})
endif()
add_library(${PROJECT_NAME}_utils SHARED ${UTILS_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_utils)
target_compile_definitions(${PROJECT_NAME}_utils PUBLIC ${BUILD_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}_utils
    yaml-cpp
    fmt::fmt
    Eigen3::Eigen
    Ceres::ceres
    spdlog::spdlog
    TBB::tbb
    Boost::boost
    ${OpenCV_LIBS}
    ${BUILD_LIBS}
    
)
file(GLOB_RECURSE AUTO_AIM_SOURCES  src/tasks/auto_aim/*.cpp)
add_library(${PROJECT_NAME}_auto_aim SHARED ${AUTO_AIM_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_auto_aim)
target_link_libraries(${PROJECT_NAME}_auto_aim
     ${PROJECT_NAME}_utils
)
file(GLOB_RECURSE AUTO_BUFF_SOURCES  src/tasks/auto_buff/*.cpp)
add_library(${PROJECT_NAME}_auto_buff SHARED ${AUTO_BUFF_SOURCES})
awakening_enable_pch(${PROJECT_NAME}_auto_buff)
target_link_libraries(${PROJECT_NAME}_auto_buff
     ${PROJECT_NAME}_utils
)
add_library(${PROJECT_NAME}_web SHARED src/tasks/base/web.cpp)
awakening_enable_pch(${PROJECT_NAME}_web)
target_link_libraries(${PROJECT_NAME}_web
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
)
if(RERUN_FOUND)
    # The distributed C++ SDK is static/non-PIC, so keep it at executable link scope.
    add_library(${PROJECT_NAME}_rerun STATIC src/_rerun/recorder.cpp)
    awakening_enable_pch(${PROJECT_NAME}_rerun)
    target_compile_definitions(${PROJECT_NAME}_rerun PUBLIC USE_RERUN)
    target_link_libraries(${PROJECT_NAME}_rerun
        PUBLIC rerun_sdk
        PRIVATE ${PROJECT_NAME}_auto_aim ${PROJECT_NAME}_auto_buff
    )
endif()

set(ROS2_PACKAGES
    ament_cmake
    rmw
    rosidl_typesupport_cpp
    rosidl_typesupport_interface
    rclcpp
    geometry_msgs
    tf2_ros
    tf2_geometry_msgs
    sensor_msgs
    visualization_msgs
    nav_msgs
    
)
foreach(pkg IN LISTS ROS2_PACKAGES)
    find_package(${pkg} QUIET)
endforeach()


set(ROS2_FULL_FOUND TRUE)
foreach(pkg IN LISTS ROS2_PACKAGES)
    if(NOT ${pkg}_FOUND)
        set(ROS2_FULL_FOUND FALSE)
        message(WARNING "ROS2 package ${pkg} not found")
    endif()
endforeach()

if(ROS2_FULL_FOUND)
    message(STATUS "ROS2 full environment found, compiling ROS2 targets ...")
    list(APPEND BUILD_DEFINITIONS USE_ROS2)

    set(ROS2_INCLUDES)
    foreach(pkg IN LISTS ROS2_PACKAGES)
        if(TARGET ${pkg}::${pkg})
            list(APPEND ROS2_INCLUDES ${${pkg}_INCLUDE_DIRS})
        endif()
    endforeach()
    include_directories(${ROS2_INCLUDES})
    file(GLOB_RECURSE RCL_SOURCES  src/_rcl/*.cpp)
    add_library(${PROJECT_NAME}_rcl SHARED EXCLUDE_FROM_ALL ${RCL_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_rcl)
    target_link_libraries(${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
    )
    ament_target_dependencies(${PROJECT_NAME}_rcl  ${ROS2_PACKAGES})
    target_compile_definitions(${PROJECT_NAME}_rcl  PUBLIC ${BUILD_DEFINITIONS})
    if(BUILD_SENTRY)
    file(GLOB_RECURSE SENTRY_BRAIN_SOURCES  src/tasks/sentry_brain/*.cpp)
    add_library(${PROJECT_NAME}_sentry_brain SHARED EXCLUDE_FROM_ALL ${SENTRY_BRAIN_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_sentry_brain)
    target_link_libraries(${PROJECT_NAME}_sentry_brain
        ${PROJECT_NAME}_rcl
        ${PROJECT_NAME}_auto_aim 
    )
    endif()
    if(BUILD_RADAR)
    # find_package(small_gicp REQUIRED)
    find_package(Open3D REQUIRED)
    file(GLOB_RECURSE RADAR_DETECT_SOURCES  src/tasks/radar_detect/*.cpp)
    add_library(${PROJECT_NAME}_radar_detect SHARED EXCLUDE_FROM_ALL ${RADAR_DETECT_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_radar_detect)
    target_link_libraries(${PROJECT_NAME}_radar_detect
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
        Open3D::Open3D
    )
    file(GLOB_RECURSE RADAR_IO_SOURCES  src/tasks/radar_io/*.cpp)
    add_library(${PROJECT_NAME}_radar_io SHARED EXCLUDE_FROM_ALL ${RADAR_IO_SOURCES})
    awakening_enable_pch(${PROJECT_NAME}_radar_io)
    target_link_libraries(${PROJECT_NAME}_radar_io
        ${PROJECT_NAME}_utils
        ${PROJECT_NAME}_rcl
    )
    endif()
else()
    message(WARNING "ROS2 dependencies incomplete, skipping ROS2 targets ...")
endif()


function(add_common_executable exe_name src_file)
    add_executable(${exe_name} ${src_file})
    awakening_enable_pch(${exe_name})
    target_link_libraries(${exe_name}  ${PROJECT_NAME}_utils relink)
    if(RERUN_FOUND)
        target_link_libraries(${exe_name} ${PROJECT_NAME}_rerun)
    endif()
endfunction()

function(add_tool_executable exe_name src_file)
    if(BUILD_RUNTIME_TESTS)
        add_executable(${exe_name} ${src_file})
    else()
        add_executable(${exe_name} EXCLUDE_FROM_ALL ${src_file})
    endif()
    awakening_enable_pch(${exe_name})
    target_link_libraries(${exe_name} ${PROJECT_NAME}_utils relink)
    if(RERUN_FOUND)
        target_link_libraries(${exe_name} ${PROJECT_NAME}_rerun)
    endif()
endfunction()

add_common_executable(standard src/runtime/standard.cpp)
add_tool_executable(control_test src/runtime/test/control_test.cpp)
add_tool_executable(delay_test src/runtime/test/delay_test.cpp)
add_tool_executable(rune_cb5_test src/runtime/test/rune_cb5_test.cpp)


target_link_libraries(standard
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
    ${PROJECT_NAME}_web
)
target_link_libraries(control_test
    ${PROJECT_NAME}_web
)
target_link_libraries(rune_cb5_test
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
)


if(BUILD_RADAR)
add_tool_executable(radar_detect src/runtime/radar_detect.cpp)
target_link_libraries(radar_detect
    ${PROJECT_NAME}_radar_detect
    ${PROJECT_NAME}_radar_io
)
add_tool_executable(livox_v1_driver src/runtime/livox_v1_driver.cpp)
target_link_libraries(livox_v1_driver
    ${PROJECT_NAME}_radar_detect
)
add_tool_executable(radar_cal src/runtime/radar_cal.cpp)
target_link_libraries(radar_cal
    ${PROJECT_NAME}_radar_detect
)
endif()
if(ROS2_FULL_FOUND)
# add_tool_executable(camera_to_ros2 src/runtime/test/camera_to_ros2.cpp)
# target_link_libraries(camera_to_ros2
#     ${PROJECT_NAME}_rcl
# )
target_link_libraries(standard
    ${PROJECT_NAME}_rcl
)

if(BUILD_SENTRY)
add_tool_executable(nav src/runtime/nav.cpp)
add_common_executable(sentry src/runtime/sentry.cpp)
add_tool_executable(map_edit src/runtime/map_edit.cpp)

target_link_libraries(nav
    ${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
)
target_link_libraries(sentry
    ${PROJECT_NAME}_rcl
    ${PROJECT_NAME}_utils
    ${PROJECT_NAME}_auto_aim
    ${PROJECT_NAME}_auto_buff
    ${PROJECT_NAME}_web
    ${PROJECT_NAME}_sentry_brain
)

endif()
endif()
