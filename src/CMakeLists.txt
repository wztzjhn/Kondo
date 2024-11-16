add_library(kondo_base OBJECT toml_helper.cpp model.cpp mostovoy_model.cpp dynamics.cpp)

add_executable(test_kondo test_kondo.cpp $<TARGET_OBJECTS:kondo_base>)
if(WITH_CUDA)
    set_target_properties(test_kondo PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86")
    target_link_libraries(test_kondo PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_link_options(test_kondo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xlinker=-lpthread -Xlinker=-lrt -Xlinker=-ldl>)
endif()
target_link_libraries(test_kondo PRIVATE ${LIBS})

add_executable(kondo kondo.cpp $<TARGET_OBJECTS:kondo_base>)
if(WITH_CUDA)
    set_target_properties(kondo PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86")
    target_link_libraries(kondo PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_link_options(kondo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xlinker=-lpthread -Xlinker=-lrt -Xlinker=-ldl>)
endif()
target_link_libraries(kondo PRIVATE ${LIBS})

add_subdirectory(extra)
