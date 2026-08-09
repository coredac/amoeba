add_executable(mlir-amoeba-opt mlir-amoeba-opt.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRTaskflowTransforms
        MLIRAmoebaConversion
        MLIRNeura
        MLIRNeuraTransforms
        MLIRNeuraConversion
        MLIRTaskflow
        MLIRTosaDialect
        MLIRTosaTransforms
        MLIRLinalgTransforms
        MLIRArithTransforms
        MLIRSCFTransforms
        MLIRTensorTransforms
        MLIRBufferizationDialect
        MLIRBufferizationTransforms
        MLIRFuncAllExtensions
        MLIRTensorAllExtensions
        MLIRTransforms
        MLIROptLib
        MLIRPass
        MLIRIR
        MLIRParser
        MLIRSupport
        )

target_link_libraries(mlir-amoeba-opt PRIVATE ${LIBS})