"""
ViT explainers (CLS + positional embeddings aware)

Included:
- _get_patch_hw(model, x): robust patch grid inference
- _collect_attn_all_layers(model, x_detached): gather per-layer attentions
- _routing_alpha(...): routing distribution over patches ('attn' | 'rollout' | 'uniform')

Explainers:
- explain_vit_with_cls_pos(...): one-backward decomposition (Euler-style), routes CLS/pos via attention/rollout; conservative (sum ≈ logit)
- ig_explain_vit_with_cls_pos(...): Integrated Gradients along joint path f(tx,tc,tp); conservative (sum ≈ f(x,c,p)-f(0,0,0))
- conditional_euler_explain_x(...): image-only Euler for Δx = f(x,c,p) - f(0,c,p); conservative for Δx when model is ~1-homogeneous in x

Notes:
- Attention is used as a *router* to spatialize non-spatial CLS/pos totals; image pixels are attributed via grad/IG.
- For strict exactness regardless of homogeneity, prefer the IG variant and increase `steps`.
"""

import math
import torch
import torch.nn.functional as F

# bcos imports
from bcos.models.vit import *                                                                 
#from bcos.models.vit_with_conv_stem import *                                                 
import sys                                                                                    
sys.path.insert(0, r'/BS/dnn_interpretablity_robustness_representation_learning/work/my_projects/bcos_dino')

__all__ = [
    "_get_patch_hw",
    "_collect_attn_all_layers",
    "_routing_alpha",
    "explain_vit_with_cls_pos",
    "ig_explain_vit_with_cls_pos",
    "conditional_euler_explain_x",
]

# ---------- Patch grid inference ----------
@torch.no_grad()
def _get_patch_hw(model, x):
    """
    Robustly get (H_p, W_p):
    - If model.prepare_tokens(x) -> (tokens, (H_p,W_p)): use it.
    - Else, use model.patch_embed.patch_size and input H,W.
    - Else, infer from pos_embed length (minus num_extra_tokens) assuming square.
    """
    out = model.prepare_tokens(x)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        _, patch_hw = out
        return patch_hw

    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'patch_size'):
        ps = model.patch_embed.patch_size
        ph, pw = (ps if isinstance(ps, (tuple, list)) else (ps, ps))
        H, W = x.shape[-2:]
        return (H // ph, W // pw)

    if hasattr(model, 'pos_embed') and isinstance(model.pos_embed, torch.Tensor):
        num_extra = int(getattr(
            model, 'num_extra_tokens',
            1 if getattr(model, 'use_cls_token', False) else 0
        ))
        L = model.pos_embed.shape[1] - num_extra
        s = int(round(math.sqrt(L)))
        if s * s != L:
            raise ValueError(f"Cannot infer patch grid from pos_embed (L={L}).")
        return (s, s)

    raise ValueError("Could not determine (H_p, W_p) for this model.")


# ---------- Collect attentions from all Transformer blocks ----------
@torch.no_grad()
def _collect_attn_all_layers(model, x_detached):
    """
    Returns a list [A^1, A^2, ..., A^L], each of shape (B, H, N, N), by hooking the Attention modules.
    Assumes each Transformer block exposes .attn with forward returning (out, attn).
    """
    attns, handles = [], []

    def hook_attn(mod, inp, out):
        # 'out' is (to_out_tensor, attn) from your Attention.forward
        attns.append(out[1].detach())

    for blk in getattr(model, 'transformer', []):
        if hasattr(blk, 'attn'):
            handles.append(blk.attn.register_forward_hook(hook_attn))

    _ = model(x_detached)  # one forward pass to fill attns

    for h in handles:
        h.remove()
    return attns  # list length == depth, each (B, heads, N, N)


# ---------- Build routing α over patches ----------
@torch.no_grad()
def _routing_alpha(
    model,
    x_detached,
    mode="attn",                    # 'attn' | 'rollout' | 'uniform'
    num_extra=1,
    rollout_add_residual=True,      # add identity before normalization
    rollout_head_reduce="mean",     # 'mean' or 'max'
    rollout_start_layer=0           # drop first k layers if desired
):
    """
    Returns alpha of shape (N_patches,), a non-negative distribution that sums to 1.
    """
    if mode == "uniform":
        H_p, W_p = _get_patch_hw(model, x_detached)
        Np = H_p * W_p
        return torch.full((Np,), 1.0 / Np, device=x_detached.device)

    if mode == "attn":
        attn = model.get_last_selfattention(x_detached)          # (B, heads, N, N)
        A = attn.mean(dim=1)[0]                                  # (N, N)
        alpha = A[0, num_extra:]                                 # CLS row -> patches
        alpha = alpha / (alpha.sum() + 1e-12)
        return alpha

    if mode == "rollout":
        attn_list = _collect_attn_all_layers(model, x_detached)
        A_bar = None
        I = None
        for A in attn_list[rollout_start_layer:]:
            A = A[0]  # (H, N, N)
            if rollout_head_reduce == "mean":
                A = A.mean(dim=0)          # (N, N)
            elif rollout_head_reduce == "max":
                A, _ = A.max(dim=0)        # (N, N)
            else:
                raise ValueError("rollout_head_reduce must be 'mean' or 'max'.")

            if I is None:
                I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
            if rollout_add_residual:
                A = A + I

            # Row-normalize
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
            A_bar = A if A_bar is None else (A_bar @ A)

        if A_bar is None:
            raise RuntimeError("No attentions collected for rollout (check model.transformer).")

        alpha = A_bar[0, num_extra:]  # CLS row -> patches
        alpha = alpha / (alpha.sum() + 1e-12)
        return alpha

    raise ValueError("Unknown routing mode. Use 'attn', 'rollout', or 'uniform'.")


# ---------- Main explainer with selectable routing (Euler-style) ----------
def explain_vit_with_cls_pos(
    model,
    x,                                # (1, C, H, W)
    y_index=None,
    redistribute='attn',              # 'attn' | 'rollout' | 'uniform'
    scale_joint=10.0,                 # scale (x,c,p) jointly to reduce LN-eps drift
    atol=5e-4, rtol=2e-2,
    return_parts=False,
    # rollout options:
    rollout_add_residual=True,
    rollout_head_reduce="mean",
    rollout_start_layer=0
):
    """
    One-backward decomposition for a ViT with CLS + additive pos_embed.
    - Computes scalar terms Sx, Sc, Sp via input×grad at the point (with joint scaling).
    - Routes (Sc + pos_CLS) via 'redistribute' to patches; pos_patch stays on its patch.
    - Returns a pixel map whose sum ≈ the chosen scalar output (conservative within tolerance).
    """
    assert x.shape[0] == 1, "Use B=1 for attribution."
    device = next((p.device for p in model.parameters()), x.device)
    H, W = x.shape[-2:]

    # ---- leaf input for attribution ----
    x0 = x.to(device).detach().requires_grad_(True)                 # LEAF
    x_fwd = x0 * float(scale_joint) if scale_joint != 1.0 else x0   # NON-LEAF for forward

    # ---- explanation-friendly mode ----
    model.eval()
    for m in model.modules():
        if hasattr(m, "explanation_mode"):    m.explanation_mode(True)
        if hasattr(m, "set_explanation_mode"): m.set_explanation_mode(True)
        if hasattr(m, "detach"):              m.detach = True
        if hasattr(m, "detach_var"):          m.detach_var = True

    # ---- temporarily enable grads on token-like params ----
    cls_token = getattr(model, 'cls_token', None)
    pos_embed = getattr(model, 'pos_embed', None)

    saved_flags, saved_data = {}, {}
    def _prep_param(name, t):
        if not isinstance(t, torch.Tensor):
            return
        saved_flags[name] = t.requires_grad
        t.requires_grad_(True)
        t.retain_grad()
        if scale_joint != 1.0:
            saved_data[name] = t.data.clone()
            with torch.no_grad():
                t.data.mul_(float(scale_joint))  # joint-scale constants (no grad)

    _prep_param('cls_token', cls_token)
    _prep_param('pos_embed', pos_embed)

    # ---- forward -> scalar ----
    model.zero_grad(set_to_none=True)
    y = model(x_fwd)
    y = y[-1] if isinstance(y, (list, tuple)) else y
    y0 = y[0]
    if y_index is None:
        y_index = torch.randint(y0.numel(), (1,), device=device).item()
    y_scalar = y0.reshape(-1)[y_index]
    y_scalar.backward()

    # ---- scalar contributions ----
    x_term = (x0 * x0.grad).sum()

    c_term = torch.tensor(0.0, device=device)
    if isinstance(cls_token, torch.Tensor) and cls_token.grad is not None:
        c_term = (cls_token * cls_token.grad).sum()

    p_term = torch.tensor(0.0, device=device)
    p_cls = torch.tensor(0.0, device=device)
    p_patches = None
    if isinstance(pos_embed, torch.Tensor) and pos_embed.grad is not None:
        pos_token = (pos_embed * pos_embed.grad).sum(dim=-1).squeeze(0)  # (L,)
        num_extra = int(getattr(model, 'num_extra_tokens',
                        1 if getattr(model, 'use_cls_token', False) else 0))
        if num_extra == 1 and pos_token.numel() >= 1:
            p_cls = pos_token[0]
            p_patches = pos_token[1:]                                   # (N_param_patches,)
        else:
            p_patches = pos_token
        p_term = pos_token.sum()

    # ---- routing α and patch map ----
    with torch.no_grad():
        H_p, W_p = _get_patch_hw(model, x_fwd.detach())
        Np = H_p * W_p
        num_extra = int(getattr(model, 'num_extra_tokens',
                        1 if getattr(model, 'use_cls_token', False) else 0))

        alpha = _routing_alpha(
            model, x_fwd.detach(), mode=redistribute, num_extra=num_extra,
            rollout_add_residual=rollout_add_residual,
            rollout_head_reduce=rollout_head_reduce,
            rollout_start_layer=rollout_start_layer
        )  # (N_patches,)
        alpha = alpha / (alpha.sum() + 1e-12)

        patch_map = torch.zeros((Np,), device=device)
        patch_map += (c_term + p_cls) * alpha

        if p_patches is not None:
            # If the pos_embed param grid doesn't match current Np, resize conservatively
            if p_patches.numel() != Np:
                s = int(round(math.sqrt(p_patches.numel())))
                grid = p_patches.view(s, s)
                up = F.interpolate(grid[None, None, :, :], size=(H_p, W_p), mode='nearest').squeeze()
                # preserve total sum after resizing
                scale = (H_p * W_p) / float(s * s)
                patch_map += up.flatten() / scale
            else:
                patch_map += p_patches

        patch_map = patch_map.view(H_p, W_p)

        # conservative upsample to pixels (sum preserved)
        patch_area = (H / float(H_p)) * (W / float(W_p))
        bias_map_px = F.interpolate(
            patch_map[None, None, :, :], size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0) / patch_area

    # image map from the LEAF x0
    img_map_px = (x0 * x0.grad).sum(dim=1).squeeze(0)  # (H, W)
    pixel_map_total = img_map_px + bias_map_px
    S_total = pixel_map_total.sum()

    # ---- conservation checks ----
    dec_ok = torch.allclose(y_scalar.detach(),
                            (x_term + c_term + p_term).detach(),
                            atol=atol, rtol=rtol)
    map_ok = torch.allclose(y_scalar.detach(),
                            S_total.detach(),
                            atol=atol, rtol=rtol)

    print(f"Logit value: {y_scalar.detach().item()}, Sum: {x_term+c_term+p_term}, x_term: {x_term}, \
            c_term: {c_term}, p_term: {p_term}")

    # ---- restore params ----
    def _restore(name, t):
        if not isinstance(t, torch.Tensor):
            return
        t.requires_grad_(saved_flags.get(name, t.requires_grad))
        if name in saved_data:
            with torch.no_grad():
                t.data.copy_(saved_data[name])
        if t.grad is not None:
            t.grad.zero_()

    _restore('cls_token', cls_token)
    _restore('pos_embed', pos_embed)

    info = {
        "ok_scalar_decomp": bool(dec_ok),
        "ok_map_conservation": bool(map_ok),
        "y_scalar": float(y_scalar.detach().cpu()),
        "sum_image_term": float(x_term.detach().cpu()),
        "cls_term": float(c_term.detach().cpu()),
        "pos_term": float(p_term.detach().cpu()),
        "sum_total_map": float(S_total.detach().cpu()),
        "y_index": int(y_index),
        "routing_mode": redistribute,
    }
    if return_parts:
        info.update({
            "pixel_map_image": img_map_px.detach().cpu(),
            "pixel_map_bias": bias_map_px.detach().cpu(),
            "patch_map_bias": patch_map.detach().cpu(),
            "alpha": alpha.detach().cpu(),
        })

    assert dec_ok and map_ok, (
        f"Sanity failed: dec_ok={dec_ok}, map_ok={map_ok}. "
        f"Try scale_joint=10–20 or relax tolerances slightly."
    )
    return pixel_map_total, info


# ---------- Integrated Gradients (joint path) ----------
def ig_explain_vit_with_cls_pos(
    model,
    x,                                # (1, C, H, W)
    y_index=None,                     # which scalar to explain; None = random index of output[0]
    steps=64,                         # #Riemann steps (increase for tighter equality)
    baseline=None,                    # None => zeros for x,c,p
    redistribute='attn',              # 'attn' | 'rollout' | 'uniform'
    atol=1e-4, rtol=1e-2,
    return_parts=False,
    # rollout options:
    rollout_add_residual=True,
    rollout_head_reduce="mean",
    rollout_start_layer=0
):
    """
    Integrated Gradients over the joint path g(t)=f(tx, tc, tp).
    Returns a conservative pixel map whose sum equals f(x,c,p) - f(0,0,0) (up to integration error).
    """
    assert x.shape[0] == 1, "Use B=1 for attribution."
    device = next((p.device for p in model.parameters()), x.device)
    H, W = x.shape[-2:]

    # ---- Choose scalar to explain (based on full input) ----
    model.eval()
    with torch.no_grad():
        y_full = model(x.to(device))
        y_full = y_full[-1] if isinstance(y_full, (list, tuple)) else y_full
        y0 = y_full[0]
        if y_index is None:
            y_index = torch.randint(y0.numel(), (1,), device=device).item()
        y_scalar_full = y0.reshape(-1)[y_index].detach()

    # ---- Prepare leaves and original constants ----
    x0 = x.to(device).detach().requires_grad_(True)  # leaf input
    cls_token = getattr(model, 'cls_token', None)
    pos_embed = getattr(model, 'pos_embed', None)

    saved_flags = {}
    saved_data  = {}
    def _prep_param(name, t):
        if not isinstance(t, torch.Tensor):
            return
        saved_flags[name] = t.requires_grad
        saved_data[name]  = t.detach().clone()
        t.requires_grad_(True)
        t.retain_grad()

    _prep_param('cls_token', cls_token)
    _prep_param('pos_embed', pos_embed)

    # Baselines (zeros by default)
    x_base = torch.zeros_like(x0) if baseline is None else baseline.to(device)
    dx = (x0 - x_base)  # (1,C,H,W)
    if cls_token is not None:
        c0   = saved_data['cls_token']
        c_base = torch.zeros_like(c0)
        dc   = (c0 - c_base)
    else:
        dc = None
    if pos_embed is not None:
        p0   = saved_data['pos_embed']
        p_base = torch.zeros_like(p0)
        dp   = (p0 - p_base)
    else:
        dp = None

    # ---- Compute baseline output f(0,0,0) ----
    with torch.no_grad():
        if cls_token is not None:
            cls_token.data.zero_()
        if pos_embed is not None:
            pos_embed.data.zero_()
        y_base = model(x_base)
        y_base = y_base[-1] if isinstance(y_base, (list, tuple)) else y_base
        y_scalar_base = y_base[0].reshape(-1)[y_index].detach()

    # ---- Integrated Gradients loop (Riemann sum) ----
    grad_x_sum = torch.zeros_like(x0)
    grad_c_sum = torch.zeros_like(cls_token) if cls_token is not None else None
    grad_p_sum = torch.zeros_like(pos_embed) if pos_embed is not None else None

    alphas = torch.linspace(1.0/steps, 1.0, steps, device=device)

    for alpha in alphas:
        # Scale inputs/params to current point on the path
        x_t = x_base + alpha * dx  # non-leaf tensor
        with torch.no_grad():
            if cls_token is not None:
                cls_token.data.copy_(c_base + alpha * dc)
            if pos_embed is not None:
                pos_embed.data.copy_(p_base + alpha * dp)

        # Forward + backward to get grads w.r.t. LEAF x0 and parameter leaves
        model.zero_grad(set_to_none=True)
        if x0.grad is not None: x0.grad.zero_()
        if cls_token is not None and cls_token.grad is not None: cls_token.grad.zero_()
        if pos_embed is not None and pos_embed.grad is not None: pos_embed.grad.zero_()

        # Re-express x_t via x0 so grads flow to x0
        x_t_for_grad = x_base + alpha * (x0 - x_base)

        y = model(x_t_for_grad)
        y = y[-1] if isinstance(y, (list, tuple)) else y
        y_scalar = y[0].reshape(-1)[y_index]
        y_scalar.backward()

        grad_x_sum += x0.grad.detach()
        if grad_c_sum is not None:
            grad_c_sum += cls_token.grad.detach()
        if grad_p_sum is not None:
            grad_p_sum += pos_embed.grad.detach()

    # Integrated Gradients: (input - baseline) * average gradient along path
    ig_x = dx * (grad_x_sum / steps)               # (1,C,H,W)
    ig_c = dc * (grad_c_sum / steps) if dc is not None else None
    ig_p = dp * (grad_p_sum / steps) if dp is not None else None

    # Restore original params
    def _restore(name, t):
        if not isinstance(t, torch.Tensor):
            return
        t.requires_grad_(saved_flags[name])
        with torch.no_grad():
            t.data.copy_(saved_data[name])
        if t.grad is not None:
            t.grad.zero_()

    if cls_token is not None: _restore('cls_token', cls_token)
    if pos_embed is not None: _restore('pos_embed', pos_embed)

    # ---- Scalar conservation check: IG sums equal f(x)-f(0) ----
    S_x   = ig_x.sum()
    S_c   = ig_c.sum() if ig_c is not None else torch.tensor(0.0, device=device)
    S_p   = ig_p.sum() if ig_p is not None else torch.tensor(0.0, device=device)
    lhs   = (S_x + S_c + S_p).detach()
    rhs   = (y_scalar_full - y_scalar_base).detach()
    dec_ok = torch.allclose(lhs, rhs, atol=atol, rtol=rtol)

    # ---- Build conservative pixel map ----
    img_map_px = ig_x.sum(dim=1).squeeze(0)  # (H, W)

    with torch.no_grad():
        H_p, W_p = _get_patch_hw(model, x.to(device))
        Np = H_p * W_p
        num_extra = int(getattr(model, 'num_extra_tokens',
                        1 if getattr(model, 'use_cls_token', False) else 0))

        alpha = _routing_alpha(
            model, x.to(device), mode=redistribute, num_extra=num_extra,
            rollout_add_residual=rollout_add_residual,
            rollout_head_reduce=rollout_head_reduce,
            rollout_start_layer=rollout_start_layer
        )
        alpha = alpha / (alpha.sum() + 1e-12)

        # Split IG(pos) into CLS vs patches and sum over D
        p_cls = torch.tensor(0.0, device=device)
        p_patches = None
        if ig_p is not None:
            token_sums = ig_p.sum(dim=-1).squeeze(0)  # (L,)
            if num_extra == 1 and token_sums.numel() >= 1:
                p_cls = token_sums[0]
                p_patches = token_sums[1:]           # (Np,)
            else:
                p_patches = token_sums

        patch_map = torch.zeros((Np,), device=device)
        patch_map += ((ig_c.sum() if ig_c is not None else 0.0) + p_cls) * alpha
        if p_patches is not None:
            if p_patches.numel() != Np:
                s = int(round(math.sqrt(p_patches.numel())))
                grid = p_patches.view(s, s)
                up = F.interpolate(grid[None, None, :, :], size=(H_p, W_p), mode='nearest').squeeze()
                scale = (H_p * W_p) / float(s * s)
                patch_map += up.flatten() / scale
            else:
                patch_map += p_patches

        patch_map = patch_map.view(H_p, W_p)

        # conservative upsample to pixels (sum preserved)
        patch_area = (H / float(H_p)) * (W / float(W_p))
        bias_map_px = F.interpolate(
            patch_map[None, None, :, :], size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0) / patch_area

    pixel_map_total = img_map_px + bias_map_px
    S_total = pixel_map_total.sum()
    map_ok = torch.allclose(S_total.detach(), (y_scalar_full - y_scalar_base).detach(),
                            atol=atol, rtol=rtol)

    info = {
        "ok_scalar_decomp": bool(dec_ok),
        "ok_map_conservation": bool(map_ok),
        "y_scalar_full": float(y_scalar_full.cpu()),
        "y_scalar_base": float(y_scalar_base.cpu()),
        "sum_image_IG": float(S_x.detach().cpu()),
        "sum_cls_IG": float(S_c.detach().cpu()),
        "sum_pos_IG": float(S_p.detach().cpu()),
        "sum_total_map": float(S_total.detach().cpu()),
        "y_index": int(y_index),
        "routing_mode": redistribute,
        "steps": int(steps),
    }
    if return_parts:
        info.update({
            "pixel_map_image": img_map_px.detach().cpu(),
            "pixel_map_bias":  bias_map_px.detach().cpu(),
            "patch_map_bias":  patch_map.detach().cpu(),
            "alpha":           alpha.detach().cpu(),
        })

    # Keep assertions optional for workflows; uncomment if desired:
    # assert dec_ok and map_ok, f"IG conservation failed: dec_ok={dec_ok}, map_ok={map_ok}"

    return pixel_map_total, info


# ---------- Conditional image-only Euler ----------
def conditional_euler_explain_x(
    model,
    x,                       # (1, C, H, W)
    y_index=None,            # pick scalar from model(x)[0]; None=random
    atol=5e-4, rtol=2e-2,    # tolerances for the conservation check
    explain_mode=True        # set your B-cos explanation toggles if available
):
    """
    Explain the image-only effect: Delta_x = f(x,c,p) - f(0,c,p), holding CLS/pos fixed.
    Returns (pixel_map, info). Sums of pixel_map ~= Delta_x if Euler is exact (1-homog in x).
    """
    assert x.shape[0] == 1, "Use B=1 for attribution."
    device = next((p.device for p in model.parameters()), x.device)
    x0 = x.to(device).detach().requires_grad_(True)
    H, W = x0.shape[-2:]

    # eval + (optional) explanation mode
    model.eval()
    if explain_mode:
        for m in model.modules():
            if hasattr(m, "explanation_mode"):   m.explanation_mode(True)
            if hasattr(m, "set_explanation_mode"): m.set_explanation_mode(True)
            if hasattr(m, "detach"):             m.detach = True
            if hasattr(m, "detach_var"):         m.detach_var = True

    # ---- choose scalar to explain ----
    with torch.no_grad():
        y_full = model(x0)
        y_full = y_full[-1] if isinstance(y_full, (list, tuple)) else y_full
        y0 = y_full[0]
        if y_index is None:
            y_index = torch.randint(y0.numel(), (1,), device=device).item()
        y_scalar_full = y0.reshape(-1)[y_index].detach()

        # baseline (x=0), but keep CLS and pos exactly as they are
        y_base = model(torch.zeros_like(x0))
        y_base = y_base[-1] if isinstance(y_base, (list, tuple)) else y_base
        y_scalar_base = y_base[0].reshape(-1)[y_index].detach()

    # ---- one backward pass at (x,c,p) ----
    model.zero_grad(set_to_none=True)
    y = model(x0)
    y = y[-1] if isinstance(y, (list, tuple)) else y
    y_scalar = y[0].reshape(-1)[y_index]
    y_scalar.backward()

    # ---- pixel map and conservation check ----
    img_map_px = (x0 * x0.grad).sum(dim=1).squeeze(0)   # (H, W)
    S_map = img_map_px.sum()
    delta_x = (y_scalar_full - y_scalar_base)

    ok = torch.allclose(S_map.detach(), delta_x.detach(), atol=atol, rtol=rtol)

    info = {
        "ok_conservation": bool(ok),
        "y_scalar_full": float(y_scalar_full.cpu()),
        "y_scalar_base": float(y_scalar_base.cpu()),
        "delta_x": float(delta_x.detach().cpu()),
        "sum_pixel_map": float(S_map.detach().cpu()),
        "y_index": int(y_index),
    }
    return img_map_px, info

if __name__ == "__main__":                                                                
                                                                                          
    no_of_runs = 20
    counts_passed = 0
    for idx in range(no_of_runs):
        curr_model = dino_like_vitc_b_patch1_14()
        try:
            test_input = torch.randn((1, 6, 224, 224))
            _, _ = explain_vit_with_cls_pos(curr_model,
                                            test_input,
                                            scale_joint=1.0,
                                            atol=1e-7,
                                            rtol=1e-8)
            counts_passed += 1
            print(f"Idx: {idx}, Passed!")
        except:
            print(f"Idx: {idx}, Failed!")
            pass

        print(f"So far, passed: {counts_passed}, failed: {idx+1-counts_passed}")

    exit(0)
