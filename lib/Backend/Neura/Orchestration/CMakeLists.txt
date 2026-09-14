add_mlir_library(MLIRAmoebaNeuraOrchestrationUtil
    orchestration_utils.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRTaskflow
    LLVMSupport
)

add_subdirectory(RoutingCriticalPathOrchestration)

add_mlir_library(MLIRAmoebaNeuraOrchestration
    Orchestration.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRAmoebaNeuraOrchestrationUtil
    MLIRAmoebaNeuraRoutingCriticalPathOrchestration
)
