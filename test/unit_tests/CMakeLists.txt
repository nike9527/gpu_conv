if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu_unit_test)
    add_subdirectory(cpu_unit_test)
endif()

if (ENABLE_GPU_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu_unit_test)
    add_subdirectory(gpu_unit_test)
endif()