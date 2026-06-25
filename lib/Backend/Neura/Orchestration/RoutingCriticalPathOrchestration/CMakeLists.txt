add_mlir_library(MLIRAmoebaNeuraRoutingCriticalPathOrchestration
    RoutingCriticalPathOrchestration.cpp

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    MLIRAmoebaNeuraOrchestrationUtil
    LLVMSupport
)
