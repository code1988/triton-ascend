#include "ascend/include/TritonToLinalg/TLAAffinityPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

class TLAAffinityToLinalgPass final
    : public PassWrapper<TLAAffinityToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TLAAffinityToLinalgPass)

  StringRef getArgument() const final { return "tla-to-linalg"; }
  StringRef getDescription() const final {
    return "Adapt explicit TLA IR to the affinity memref kernel ABI";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::FuncOp> functions;
    module.walk([&](triton::FuncOp func) { functions.push_back(func); });
    if (functions.size() != 1 || functions.front().isDeclaration() ||
        functions.front().getSymVisibility() != "public") {
      module.emitError("TLA adapter expects one public tt.func definition");
      signalPassFailure();
      return;
    }

    triton::FuncOp source = functions.front();
    if (source.getFunctionType().getNumResults() != 0 ||
        !source.getBody().hasOneBlock()) {
      source.emitError("TLA adapter requires a single-block void kernel");
      signalPassFailure();
      return;
    }

    OpBuilder builder(source);
    SmallVector<Type> inputTypes;
    SmallVector<Type> originalTypes(source.getFunctionType().getInputs());
    for (Type type : originalTypes) {
      if (auto pointer = dyn_cast<triton::PointerType>(type)) {
        if (isa<triton::PointerType, ShapedType>(pointer.getPointeeType())) {
          source.emitError("TLA kernel arguments require scalar pointees");
          signalPassFailure();
          return;
        }
        inputTypes.push_back(
            MemRefType::get({ShapedType::kDynamic}, pointer.getPointeeType()));
      } else {
        inputTypes.push_back(type);
      }
    }
    auto target = builder.create<func::FuncOp>(
        source.getLoc(), source.getName(),
        builder.getFunctionType(inputTypes,
                                source.getFunctionType().getResults()));
    SmallVector<DictionaryAttr> argAttrs, resultAttrs;
    source.getAllArgAttrs(argAttrs);
    source.getAllResultAttrs(resultAttrs);
    if (!argAttrs.empty())
      target.setAllArgAttrs(argAttrs);
    if (!resultAttrs.empty())
      target.setAllResultAttrs(resultAttrs);
    IRMapping mapping;
    source.getBody().cloneInto(&target.getBody(), mapping);
    Block &body = target.getBody().front();
    for (auto [arg, type] : llvm::zip_equal(body.getArguments(), inputTypes))
      arg.setType(type);
    for (unsigned i = 0; i < 6; ++i)
      if (failed(target.insertArgument(originalTypes.size() + i,
                                       builder.getI32Type(), nullptr,
                                       source.getLoc()))) {
        target.emitError("failed to append TLA program arguments");
        signalPassFailure();
        return;
      }

    auto fail = [&](Operation *op, StringRef message) {
      op->emitError(message);
      target.erase();
      signalPassFailure();
    };

    for (unsigned i = 0; i < originalTypes.size(); ++i) {
      auto pointer = dyn_cast<triton::PointerType>(originalTypes[i]);
      if (!pointer)
        continue;
      BlockArgument arg = body.getArgument(i);
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : arg.getUses())
        uses.push_back(&use);
      for (OpOperand *use : uses) {
        Operation *owner = use->getOwner();
        if (auto ptrToInt = dyn_cast<triton::PtrToIntOp>(owner)) {
          builder.setInsertionPoint(ptrToInt);
          Value address =
              builder.create<memref::ExtractAlignedPointerAsIndexOp>(
                  ptrToInt.getLoc(), arg);
          Value integer = builder.create<arith::IndexCastOp>(
              ptrToInt.getLoc(), ptrToInt.getType(), address);
          ptrToInt.replaceAllUsesWith(integer);
          ptrToInt.erase();
          continue;
        }
        fail(owner, "unsupported TLA pointer use outside affinity operations");
        return;
      }
    }

    // The TLA frontend emits all other affinity operations directly. Only
    // program information and the entry terminator need adapter rewrites.
    SmallVector<Operation *> tritonOps;
    target.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "tt")
        tritonOps.push_back(op);
    });
    for (Operation *op : tritonOps) {
      if (auto ret = dyn_cast<triton::ReturnOp>(op)) {
        builder.setInsertionPoint(ret);
        builder.create<func::ReturnOp>(ret.getLoc(), ret.getOperands());
        ret.erase();
      } else if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
        unsigned axis = static_cast<unsigned>(pid.getAxis());
        if (axis >= 3) {
          fail(op, "invalid TLA program ID axis");
          return;
        }
        pid.replaceAllUsesWith(
            body.getArgument(originalTypes.size() + 3 + axis));
        pid.erase();
      } else if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
        unsigned axis = static_cast<unsigned>(count.getAxis());
        if (axis >= 3) {
          fail(op, "invalid TLA program count axis");
          return;
        }
        count.replaceAllUsesWith(body.getArgument(originalTypes.size() + axis));
        count.erase();
      } else {
        fail(op, "unsupported TTIR in TLA affinity module");
        return;
      }
    }

    target->setAttr("global_kernel", builder.getStringAttr("local"));
    target->setAttr("mix_mode", builder.getStringAttr("aiv"));
    target->setAttr("parallel_mode", builder.getStringAttr("simd"));
    auto runtimeArg =
        MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
    if (failed(
            target.insertArgument(0, runtimeArg, nullptr, source.getLoc())) ||
        failed(
            target.insertArgument(1, runtimeArg, nullptr, source.getLoc()))) {
      target.emitError("failed to prepend TLA runtime arguments");
      signalPassFailure();
      return;
    }
    target->setAttr("SyncBlockLockArgIdx", builder.getI64IntegerAttr(0));
    target->setAttr("WorkspaceArgIdx", builder.getI64IntegerAttr(1));

    source.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTLAAffinityToLinalgPass() {
  return std::make_unique<TLAAffinityToLinalgPass>();
}
