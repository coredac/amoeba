add_mlir_library(MLIRAmoebaNeuraPipelineInterval
    PipelineInterval.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRTaskflow
    LLVMSupport
)
