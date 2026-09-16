add_mlir_library(MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
    AnalyticalBasedTaskOrchestration.cpp

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
