// tools/mlir-amoeba-opt/mlir-amoeba-opt.cpp

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Backend/Backends.h"
#include "Conversion/AmoebaConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowPasses.h"

int main(int argc, char **argv) {
  // Registers MLIR dialects.
  mlir::DialectRegistry registry;
  mlir::amoeba::registerBackends(registry);
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::taskflow::TaskflowDialect>();
  registry.insert<mlir::math::MathDialect>();
  mlir::registerAllExtensions(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

  mlir::registerAllPasses();
  mlir::registerAmoebaConversionPasses();
  mlir::registerViewOpGraphPass();
  mlir::taskflow::registerPasses();
  mlir::taskflow::registerTosaToAffineConversionPassPipeline();
  mlir::taskflow::registerTaskflowConversionPassPipeline();
  mlir::taskflow::registerLinalgToAffineConversionPassPipeline();

  // Register all standard conversion passes
  mlir::registerConversionPasses();

  // Runs the MLIR optimizer.
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Amoeba Dialect Optimizer", registry));
}
