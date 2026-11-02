"""Minimal TLA SIMT VF: GM load -> Tensor add -> GM store.

Run with TLA_TEST_STAGE=ir / compile / run, or leave it unset for all stages.
The case is a development target for SIMT VF lowering and device execution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (check_compile_artifacts, compile_offline,
                                  get_stages, lower_offline)


N = 128
SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32"}


@triton.jit
def gm_add_one(X_ptr, O_ptr, BLOCK: tl.constexpr):
    with tla.scope.vector():
        with tla.scope.simt():
            offsets = tl.arange(0, BLOCK)
            values = tl.load(X_ptr + offsets)
            tl.store(O_ptr + offsets, values + 1.0)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gm_add_one, SIGNATURE, {"BLOCK": N}, output_dir=output_dir)
    text = str(module)
    assert "%X_ptr: !tt.ptr<f32>" in text and "%O_ptr: !tt.ptr<f32>" in text
    assert text.count("tt.load") == 1 and text.count("tt.store") == 1
    assert "#hivm.vf_mode<SIMT>" in text
    assert "memref.alloc" not in text and "hivm.hir.load" not in text

    lowered = str(lower_offline(module, ctx, output_dir=output_dir))
    assert "func.func @gm_add_one" in lowered
    assert "#hivm.vf_mode<SIMT>" in lowered
    assert "memref<?xf32>" in lowered
    assert lowered.count("tt.load") == 1 and lowered.count("tt.store") == 1
    assert lowered.count("builtin.unrealized_conversion_cast") == 2
    assert "arith.constant dense<1.000000e+00> : tensor<128xf32>" in lowered
    simt_body = lowered.split("scope.scope : () -> () {", 2)[2].split("scope.return", 1)[0]
    assert simt_body.count("builtin.unrealized_conversion_cast") == 2
    assert simt_body.count("tt.load") == 1 and simt_body.count("tt.store") == 1
    for forbidden in ("memref.alloc", "memref.copy", "bufferization.materialize_in_destination"):
        assert forbidden not in lowered, forbidden
    print("IR PASS: GM -> SIMT Tensor -> GM")


def _launch():
    import torch

    x = torch.arange(N, dtype=torch.float32, device="npu") / 8
    out = torch.empty_like(x)
    gm_add_one[(1,)](x, out, BLOCK=N, num_warps=4, enable_tla_mode=True)
    torch.npu.synchronize()
    return x, out


def test_compile():
    _launch()
    assert check_compile_artifacts(), "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: gm_add_one")


def test_run():
    import torch

    x, out = _launch()
    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    print("GM_ADD_ONE OK")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gm_add_one", "ir" if do_ir else "", "compile" if do_compile else "",
          "run" if do_run else "")


if __name__ == "__main__":
    main()
