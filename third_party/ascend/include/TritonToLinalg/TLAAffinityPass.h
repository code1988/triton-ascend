#ifndef TRITON_ASCEND_TLA_AFFINITY_PASS_H
#define TRITON_ASCEND_TLA_AFFINITY_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTLAAffinityToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ASCEND_TLA_AFFINITY_PASS_H
