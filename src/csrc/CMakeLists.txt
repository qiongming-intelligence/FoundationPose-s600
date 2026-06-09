add_library(foundationpose_s600
    bpu/bpu_allocator.cpp
    bpu/bpu_model.cpp
)

add_library(foundationpose_s600::foundationpose_s600 ALIAS foundationpose_s600)

target_include_directories(foundationpose_s600
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(foundationpose_s600 PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(foundationpose_s600 PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(FOUNDATIONPOSE_S600_ENABLE_HOBOT AND EXISTS "/usr/hobot/include")
    target_include_directories(foundationpose_s600 PUBLIC /usr/hobot/include)
    target_compile_definitions(foundationpose_s600 PRIVATE FOUNDATIONPOSE_S600_HAS_HOBOT=1)
endif()

if(FOUNDATIONPOSE_S600_ENABLE_HOBOT AND (EXISTS "/usr/include/hobot/dnn/hb_dnn.h" OR EXISTS "/usr/hobot/include/hobot/dnn/hb_dnn.h") AND EXISTS "/usr/hobot/lib/libdnn.so")
    target_compile_definitions(foundationpose_s600 PRIVATE FOUNDATIONPOSE_S600_HAS_HOBOT_DNN=1)
    target_link_libraries(foundationpose_s600 PUBLIC /usr/hobot/lib/libdnn.so)
    if(EXISTS "/usr/hobot/lib/libhbucp.so")
        target_link_libraries(foundationpose_s600 PUBLIC /usr/hobot/lib/libhbucp.so)
    endif()
endif()

if(FOUNDATIONPOSE_S600_ENABLE_HOBOT AND (EXISTS "/usr/include/hb_bpu_mem.h" OR EXISTS "/usr/hobot/include/hb_bpu_mem.h") AND EXISTS "/usr/hobot/lib/libbpu.so")
    target_compile_definitions(foundationpose_s600 PRIVATE FOUNDATIONPOSE_S600_HAS_HOBOT_BPU_MEM=1)
    target_link_libraries(foundationpose_s600 PUBLIC /usr/hobot/lib/libbpu.so)
endif()

if(FOUNDATIONPOSE_S600_BUILD_APPS)
    add_executable(foundationpose_bpu_runner tools/foundationpose_bpu_runner.cpp)
    target_link_libraries(foundationpose_bpu_runner PRIVATE foundationpose_s600::foundationpose_s600)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(foundationpose_bpu_runner PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    add_library(foundationpose_placeholder_renderer SHARED tools/placeholder_renderer.cpp)
    set_target_properties(foundationpose_placeholder_renderer PROPERTIES
        OUTPUT_NAME foundationpose_placeholder_renderer
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(foundationpose_placeholder_renderer PRIVATE cxx_std_17)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(foundationpose_placeholder_renderer PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(foundationpose_placeholder_renderer PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -O3)
    endif()
endif()

if(FOUNDATIONPOSE_S600_ENABLE_HOBOT)
    add_library(foundationpose_bpu_core1_preload SHARED tools/bpu_core1_preload.cpp)
    set_target_properties(foundationpose_bpu_core1_preload PROPERTIES
        OUTPUT_NAME foundationpose_bpu_core1_preload
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(foundationpose_bpu_core1_preload PRIVATE cxx_std_17)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(foundationpose_bpu_core1_preload PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endif()
