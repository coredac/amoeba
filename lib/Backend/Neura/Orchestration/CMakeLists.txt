add_mlir_library(MLIRAmoebaNeuraOrchestrationUtil
    orchestration_utils.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRPass
    MLIRSupport
    MLIRFuncDialect
    MLIRNeura
    MLIRTaskflow
    LLVMSupport
)

add_subdirectory(AnalyticalBasedTaskOrchestration)
add_subdirectory(RoutingCriticalPathOrchestration)

add_mlir_library(MLIRAmoebaNeuraOrchestration
    Orchestration.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRAmoebaNeuraBackendPassesIncGen

    LINK_LIBS PUBLIC
    MLIRAmoebaNeuraOrchestrationUtil
    MLIRAmoebaNeuraRoutingCriticalPathOrchestration
    MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
)
