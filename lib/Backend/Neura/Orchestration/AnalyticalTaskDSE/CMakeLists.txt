add_mlir_library(MLIRAmoebaNeuraAnalyticalTaskDSE
    SpatialDSEOrchestration.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRAmoebaNeuraOrchestrationUtil
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTaskflow
    LLVMSupport
)
