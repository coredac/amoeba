add_subdirectory(TaskflowToNeura)

add_library(MLIRAmoebaNeuraConversion INTERFACE)

add_dependencies(
  MLIRAmoebaNeuraConversion
  MLIRAmoebaNeuraBackendPassesIncGen
)

target_link_libraries(MLIRAmoebaNeuraConversion INTERFACE
  MLIRAmoebaTaskflowToNeura
)
