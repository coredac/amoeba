get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(AffineToTaskflow)
add_subdirectory(TaskflowToNeura)

add_library(MLIRAmoebaConversion INTERFACE)

add_dependencies(MLIRAmoebaConversion MLIRAmoebaConversionIncGen)

target_link_libraries(MLIRAmoebaConversion INTERFACE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRAffineToTaskflowPass
  MLIRTaskflowToNeuraPass
  ${dialect_libs}
)