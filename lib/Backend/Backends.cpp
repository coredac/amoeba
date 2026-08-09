//===- Backends.cpp - Amoeba backend registration ------------------------===//

#include "Backend/Backends.h"
#include "Backend/Neura/NeuraBackend.h"

void mlir::amoeba::registerBackends(DialectRegistry &registry) {
  registerNeuraBackend(registry);
}
