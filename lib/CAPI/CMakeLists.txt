add_mlir_public_c_api_library(MLIRTaskflowCAPI
  Taskflow.cpp

  LINK_LIBS PUBLIC
  MLIRTaskflow
)
