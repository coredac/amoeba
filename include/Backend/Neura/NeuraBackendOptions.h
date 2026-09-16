//===- NeuraBackendOptions.h - shared Neura CLI options --------*- C++ -*-===//

#ifndef AMOEBA_BACKEND_NEURA_NEURABACKENDOPTIONS_H
#define AMOEBA_BACKEND_NEURA_NEURABACKENDOPTIONS_H

#include <string>

namespace mlir {
namespace amoeba {

// These accessors live in a dependency-neutral library shared by the backend
// entry point and its pass libraries.
const std::string &getNeuraArchitectureSpecFile();
const std::string &getNeuraLatencySpecFile();

} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_NEURABACKENDOPTIONS_H
