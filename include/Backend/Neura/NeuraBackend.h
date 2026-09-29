//===- NeuraBackend.h - Neura backend registration ------------*- C++ -*-===//

#ifndef AMOEBA_BACKEND_NEURA_NEURABACKEND_H
#define AMOEBA_BACKEND_NEURA_NEURABACKEND_H

#include <string>

namespace mlir {
class DialectRegistry;

namespace amoeba {

// Sets the architecture specification before a Neura backend pass runs.
void setNeuraArchitectureSpec(const std::string &architectureSpec);

// Registers the Neura dialect, passes, and backend-specific options.
void registerNeuraBackend(DialectRegistry &registry);

} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_NEURABACKEND_H
