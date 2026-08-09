//===- NeuraBackend.h - Neura backend registration ------------*- C++ -*-===//

#ifndef AMOEBA_BACKEND_NEURA_NEURABACKEND_H
#define AMOEBA_BACKEND_NEURA_NEURABACKEND_H

namespace mlir {
class DialectRegistry;

namespace amoeba {

// Registers the Neura dialect, passes, and backend-specific options.
void registerNeuraBackend(DialectRegistry &registry);

} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_NEURABACKEND_H
