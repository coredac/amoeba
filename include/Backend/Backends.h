//===- Backends.h - Amoeba backend registration ----------------*- C++ -*-===//

#ifndef AMOEBA_BACKEND_BACKENDS_H
#define AMOEBA_BACKEND_BACKENDS_H

namespace mlir {
class DialectRegistry;

namespace amoeba {

// Registers every backend enabled in this Amoeba build.
void registerBackends(DialectRegistry &registry);

} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_BACKENDS_H
