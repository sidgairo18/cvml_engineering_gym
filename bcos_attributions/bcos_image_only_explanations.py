import torch

"""
Image-only explanation + percentages

Use the conditional explainer we discussed (Euler with three modes). For best faithfulness without heavy cost, start with mode="scaled_point", t_eval=10.0, or for guarantees, use mode="riemann_k", K=8.

How to present it in a table:
+ y_full = f(x,c,p)
+ y_base = f(0,c,p) (“rest of the model”, i.e., CLS/pos + architectural constants)
+ Δx = y_full − y_base (what your heatmap explains)
+ ∑M_img and Residual % (how close ∑heatmap is to Δx)
+ Optionally, “image share vs full logit” (signed %).
"""

# ---- helper for residuals & shares ----
def _residual_and_weight(name, map_sum_t, target_t, parts=None, eps=1e-12):
    target = float(target_t.detach().cpu() if hasattr(target_t, "detach") else target_t)
    map_sum = float(map_sum_t.detach().cpu() if hasattr(map_sum_t, "detach") else map_sum_t)
    denom = abs(target) if abs(target) > eps else eps
    report = {
        "name": name,
        "target": target,
        "map_sum": map_sum,
        "residual": map_sum - target,
        "residual_pct_abs": abs(map_sum - target) / denom * 100.0,
        "residual_pct_signed": ((map_sum - target) / (target if abs(target) > eps else eps)) * 100.0,
    }
    if parts:
        abs_sum_parts = sum(abs(float(v.detach().cpu() if hasattr(v, "detach") else v)) for v in parts.values())
        abs_sum_parts = abs_sum_parts if abs_sum_parts > eps else eps
        for k, v in parts.items():
            v_f = float(v.detach().cpu() if hasattr(v, "detach") else v)
            report[f"{k}_share_pct_signed_vs_target"] = (v_f / (target if abs(target) > eps else eps)) * 100.0
            report[f"{k}_share_pct_abs_vs_parts"] = (abs(v_f) / abs_sum_parts) * 100.0
    return report

# ---- conditional image-only explainer (Δx = f(x,c,p) - f(0,c,p)) ----
def conditional_euler_explain_x(model, x, y_index=None,
                                mode="scaled_point", t_eval=10.0, K=8,
                                atol=5e-4, rtol=2e-2, explain_mode=True):
    assert x.shape[0] == 1
    device = next((p.device for p in model.parameters()), x.device)
    x0 = x.to(device).detach().requires_grad_(True)
    H, W = x0.shape[-2:]

    # put model into evaluation/explanation mode
    model.eval()
    if explain_mode:
        for m in model.modules():
            if hasattr(m, "explanation_mode"):   m.explanation_mode(True)
            if hasattr(m, "set_explanation_mode"): m.set_explanation_mode(True)
            if hasattr(m, "detach"):             m.detach = True
            if hasattr(m, "detach_var"):         m.detach_var = True

    # choose scalar
    with torch.no_grad():
        y_full = model(x0); y_full = y_full[-1] if isinstance(y_full, (list, tuple)) else y_full
        y0 = y_full[0]
        if y_index is None:
            y_index = torch.randint(y0.numel(), (1,), device=device).item()
        y_scalar_full = y0.reshape(-1)[y_index].detach()

        # baseline keeps CLS/pos fixed, only x -> 0
        y_base = model(torch.zeros_like(x0)); y_base = y_base[-1] if isinstance(y_base, (list, tuple)) else y_base
        y_scalar_base = y_base[0].reshape(-1)[y_index].detach()

    delta_x = (y_scalar_full - y_scalar_base)

    # single helper: gradient at scale t (with chain correction 1/t)
    def grad_at_t(t: float):
        x_leaf = x0.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        y = model(t * x_leaf); y = y[-1] if isinstance(y, (list, tuple)) else y
        y_scalar = y[0].reshape(-1)[y_index]
        y_scalar.backward()
        return (x_leaf.grad / float(t)).detach()

    if mode == "point":
        model.zero_grad(set_to_none=True)
        y = model(x0); y = y[-1] if isinstance(y, (list, tuple)) else y
        y_scalar = y[0].reshape(-1)[y_index]; y_scalar.backward()
        grad = x0.grad.detach()
    elif mode == "scaled_point":
        assert t_eval > 0
        grad = grad_at_t(t_eval)
    elif mode == "riemann_k":
        K = max(2, int(K))
        alphas = torch.linspace(1.0 / K, 1.0, K, device=device)
        grad = torch.stack([grad_at_t(float(a)) for a in alphas], dim=0).mean(dim=0)
    else:
        raise ValueError("mode must be 'point', 'scaled_point', or 'riemann_k'.")

    img_map_px = (x0 * grad).sum(dim=1).squeeze(0)  # (H, W)
    S_map = img_map_px.sum()

    ok = torch.allclose(S_map.detach(), delta_x.detach(), atol=atol, rtol=rtol)
    info = {
        "ok_conservation": bool(ok),
        "mode": mode, "t_eval": (float(t_eval) if mode=="scaled_point" else None), "K": (int(K) if mode=="riemann_k" else None),
        "y_scalar_full": float(y_scalar_full.cpu()),
        "y_scalar_base": float(y_scalar_base.cpu()),
        "delta_x": float(delta_x.detach().cpu()),
        "sum_pixel_map": float(S_map.detach().cpu()),
        "y_index": int(y_index),
    }
    # percentages you can report
    info["residual_report"] = _residual_and_weight(
        name="image_only", map_sum_t=S_map, target_t=delta_x, parts={"image": S_map}
    )
    # (Optional) image share relative to the full logit:
    # be careful when y_full ~ 0; this is just for presentation
    eps = 1e-12
    y_full_val = info["y_scalar_full"]
    denom = y_full_val if abs(y_full_val) > eps else eps
    info["image_share_vs_full_signed_pct"] = float(S_map.detach().cpu()) / denom * 100.0

    return img_map_px, info

