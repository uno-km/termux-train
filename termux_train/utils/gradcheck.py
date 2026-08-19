"""
termux_train.utils.gradcheck
============================
Finite Difference Numerical Gradient Checker.
Validates analytical Autograd gradients against numerical approximations.
"""

from typing import Callable, Tuple, List
from ..tensor import Tensor

def gradcheck(
    func: Callable[..., Tensor],
    inputs: Tuple[Tensor, ...],
    eps: float = 1e-3,
    atol: float = 1e-2,
    rtol: float = 1e-2
) -> bool:
    """
    Check gradients computed via Autograd against finite difference approximations.
    
    Args:
        func: Python function taking Tensors and returning a scalar loss Tensor.
        inputs: Tuple of input Tensors with requires_grad=True.
        eps: Perturbation step size.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        
    Returns:
        True if all analytical gradients match numerical gradients within tolerance.
    """
    # 1. Compute Analytical Gradients via backward()
    for inp in inputs:
        inp.zero_grad()
        
    out = func(*inputs)
    if out.shape != ():
        raise RuntimeError(
            f"gradcheck expects func(*inputs) to return a scalar Tensor (shape ()), "
            f"but got shape {out.shape}. Reduce output with sum() or mean() before calling gradcheck."
        )
    out.backward()
    
    # 2. Compute Numerical Gradients via Finite Differences
    for arg_idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
            
        flat_data = inp.backend.to_flat_list(inp._data)
        flat_grad = inp.backend.to_flat_list(inp.grad._data) if inp.grad is not None else [0.0] * len(flat_data)
        
        num_grad = [0.0] * len(flat_data)
        shape = inp.shape
        
        for elem_idx in range(len(flat_data)):
            orig_val = flat_data[elem_idx]
            
            # f(x + eps)
            flat_data[elem_idx] = orig_val + eps
            inp._data = inp.backend.from_data(inp.backend.reshape(flat_data, shape))
            out_pos_t = func(*inputs)
            if out_pos_t.shape != ():
                raise RuntimeError("func returned non-scalar during numerical gradient step")
            out_pos = out_pos_t.item()
            
            # f(x - eps)
            flat_data[elem_idx] = orig_val - eps
            inp._data = inp.backend.from_data(inp.backend.reshape(flat_data, shape))
            out_neg_t = func(*inputs)
            if out_neg_t.shape != ():
                raise RuntimeError("func returned non-scalar during numerical gradient step")
            out_neg = out_neg_t.item()
            
            # Reset original
            flat_data[elem_idx] = orig_val
            inp._data = inp.backend.from_data(inp.backend.reshape(flat_data, shape))
            
            # Two-sided difference: (f(x+eps) - f(x-eps)) / (2*eps)
            diff = (out_pos - out_neg) / (2.0 * eps)
            num_grad[elem_idx] = diff
            
            # Check numerical vs analytical error
            analytical = flat_grad[elem_idx]
            numerical = diff
            abs_err = abs(analytical - numerical)
            denom = max(abs(analytical), abs(numerical), 1e-7)
            rel_err = abs_err / denom
            
            if abs_err > atol and rel_err > rtol:
                raise ValueError(
                    f"Gradcheck failed for input {arg_idx} at element {elem_idx}!\n"
                    f"  Analytical Grad: {analytical:.8f}\n"
                    f"  Numerical Grad:  {numerical:.8f}\n"
                    f"  Absolute Error:  {abs_err:.8e} (atol={atol})\n"
                    f"  Relative Error:  {rel_err:.8e} (rtol={rtol})"
                )
                
    return True
