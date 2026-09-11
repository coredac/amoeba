add_mlir_library(MLIRAmoebaNeuraAnalyticalTaskDSE
    SpatialShapeUtils.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTaskflow
    LLVMSupport
)
