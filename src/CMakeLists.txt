add_library(kondo_base OBJECT toml_helper.cpp model.cpp mostovoy_model.cpp dynamics.cpp)

add_executable(test_kondo test_kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(test_kondo PRIVATE ${LIBS})

add_executable(kondo kondo.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries(kondo PRIVATE ${LIBS})

add_subdirectory(extra)
