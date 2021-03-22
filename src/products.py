"""This holds functions for things like jvp/vjp and hvp"""
import torch as th


def _check_inputs_jvp(f, x, v):
    # Default to list
    if isinstance(x, th.Tensor):
        x = [x]
    if isinstance(v, th.Tensor):
        v = [v]
    if isinstance(f, th.Tensor):
        f = [f]
    # Validate the input
    if len(v) != len(x):
        raise ValueError(
            f"params and v must have the same length. "
            f"({len(v)}=/={len(x)})"
        )
    elif any(x_i.size() != v_i.size() for x_i, v_i in zip(x, v)):
        raise ValueError(
            f"Mismatched dimensions between v ({[v_i.size() for v_i in v]})"
            f" and x ({[x_i.size() for x_i in x]})"
        )
    return f, x, v


def _check_inputs_vjp(f, x, v):
    # Default to list
    if isinstance(x, th.Tensor):
        x = [x]
    if isinstance(v, th.Tensor):
        v = [v]
    if isinstance(f, th.Tensor):
        f = [f]
    # Validate the input
    if len(f) != len(v):
        raise ValueError(
            f"params and v must have the same length. "
            f"({len(v)}=/={len(f)})"
        )
    elif any(f_i.size() != v_i.size() for f_i, v_i in zip(f, v)):
        raise ValueError(
            f"Mismatched dimensions between v ({[v_i.size() for v_i in v]})"
            f" and f ({[f_i.size() for f_i in f]})"
        )
    return f, x, v


def vjp(f, x, v, retain_graph=False):
    """Vector-Jacobian product

    v^T . df/dx

    vjp(f, x, v)_{i} = \\sum_j df_j/dx_i . v_j
    """
    # Check inputs
    f, x, v = _check_inputs_vjp(f, x, v)
    # Get df/dx . v
    v_dot_df_dx = th.autograd.grad(f, x, grad_outputs=v, create_graph=True,
                                   retain_graph=retain_graph)
    # Detach
    return [v_dot_df_dx_i.clone().detach() for v_dot_df_dx_i in v_dot_df_dx]


def jvp(f, x, v, retain_graph=False):
    """Jacobian-vector product

    df/dx . v

    jvp(f, x, v)_{i} = \\sum_j df_i/dx_j . v_j
    """
    # Check inputs
    f, x, v = _check_inputs_jvp(f, x, v)
    # Dummy variable
    z = [th.zeros_like(f_i).requires_grad_(True) for f_i in f]
    # Get df/dx . z
    z_dot_df_dx = th.autograd.grad(f, x, grad_outputs=z, create_graph=True,
                                   retain_graph=retain_graph)
    # Differentiate wrt. z
    df_dx_dot_v = th.autograd.grad(
        z_dot_df_dx,
        z,
        grad_outputs=v,
        create_graph=True,
        allow_unused=True,
    )
    # Detach
    return [df_dx_dot_v_i.clone().detach() for df_dx_dot_v_i in df_dx_dot_v]


def hvp(f, x, v, retain_graph=False):
    """Hessian vector product [d^2f/dx^2] . v

    Args:
        f(torch.Tensor): tensor for the function
        x(list): input(against which we'll differentiate). Can be a list.
        v(list): vector with which the hessian is multiplied.
            Can also be a list(same length as x)
    """
    # Check inputs
    f, x, v = _check_inputs_jvp(f, x, v)
    # Also f must be a scalar
    if any(f_i.numel() > 1 for f_i in f):
        raise ValueError(
            f"f must be scalar for HVP (got {[f_i.size() for f_i in f]})"
        )
    # First backprop get df/dx
    df_dx = th.autograd.grad(f, x, create_graph=True,
                             retain_graph=retain_graph)
    # Second backprop
    # (H*v)_i = d/dx_i \sum_j df/dx_j v_j
    return th.autograd.grad(df_dx, x, grad_outputs=v,
                            retain_graph=retain_graph)
