#---------------------------------------------
# 依赖gtest
#---------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/gtest.cmake)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_unit_test)
    add_subdirectory(cpu_unit_test)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu_unit_test)
    add_subdirectory(gpu_unit_test)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu_pipeline_test)
    add_subdirectory(gpu_pipeline_test)
endif()