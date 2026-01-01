#在构建完成后，自动将目标（可执行文件）依赖的所有运行时 DLL 复制到可执行文件所在目录。
function(copy_runtime_dlls target)
    if (WIN32 AND MSVC)
        add_custom_command(TARGET ${target} POST_BUILD
            #-----------------------------------------------------------------------
            # 复制 gtest DLL
            # $<TARGET_RUNTIME_DLLS:target> 返回 target 在运行时所依赖的所有 DLL 文件路径
            # $<TARGET_FILE_DIR:${target}> 返回 target 生成文件所在的目录（exe / dll 所在目录）
            # $<TARGET_FILE:target> 返回 target 带文件完整路径
            # $<TARGET_FILE_NAME:target> 返回 target 文件名
            #-----------------------------------------------------------------------
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}>
                $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS    
        )  
    endif()
endfunction()
