"""
Phase 4: ST-PINN Three-Part Physics-Informed Loss
==================================================

Three-part loss for the Pseudo-Spatiotemporal Diffusion Model:

  L_total = L_Data + λ_eq · L_Eq + λ_pde · L_PDE

  ┌─────────────────────────────────────────────────────────────────┐
  │ L_Data = MSE(f_scaled^(K), y_scaled)                            │
  │   Standard data-fitting loss in normalized space.               │
  ├─────────────────────────────────────────────────────────────────┤
  │ L_Eq = MSE(ρ_v^(K) / σ, 0)                                     │
  │   Terminal equilibrium constraint.  Forces the final pressure   │
  │   field to zero, ensuring predicted flows satisfy implicit OD   │
  │   demand D_v — without ever needing the OD matrix.              │
  ├─────────────────────────────────────────────────────────────────┤
  │ L_PDE = mean_{k=1}^{K-1} L1(ρ_v^(k) / σ, 0)                   │
  │   Intermediate smoothing constraint.  Encourages pressure to    │
  │   decay monotonically at every pseudo-time step, not just at    │
  │   the terminal step.  Uses L1 (MAE) for robustness to outlier  │
  │   nodes with transiently high pressure during redistribution.   │
  │   Skips k=0 (initial shockwave, non-zero by definition) and    │
  │   k=K (already penalized by L_Eq with stricter MSE).            │
  └─────────────────────────────────────────────────────────────────┘

  Division by σ (flow_std) normalizes all physical-space pressures
  to dimensionless O(1), matching gradient scale of L_Data.

Key design:
  Phase 3 forward pass pre-computes ρ_v^(0..K) and attaches them to
  ``batch.rho_v_final`` and ``batch.rho_v_history``.  This loss
  simply reads the pre-computed values — gradients flow naturally
  back through the entire diffusion loop via BPTT.

Backward compatibility:
  If ``batch.rho_v_final`` is absent (baseline model), only L_Data
  is returned.  If K ≤ 1 (no intermediate steps), L_PDE = 0.

Configuration dependencies:
  cfg.dataset.flow_std   : float  - StandardScaler σ
  cfg.model.lambda_eq    : float  - Terminal equilibrium weight (default 1.0)
  cfg.model.lambda_pde   : float  - Intermediate smoothing weight (default 0.05)
"""

import torch
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg


def compute_pinn_loss(
    pred: torch.Tensor,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ST-PINN loss: L_total = L_Data + λ_eq · L_Eq + λ_pde · L_PDE.

    Args:
        pred  : [E_new, 1]  predicted equilibrium flows (normalized space)
        batch : PyG Batch object containing:
                  batch.y              : [E_new, 1]  ground truth (normalized)
                  batch.rho_v_final    : [N, 1]      terminal ρ^(K) (real, veh/hr)
                  batch.rho_v_history  : list of K+1 tensors [N, 1]
                                         ρ^(0), ρ^(1), …, ρ^(K)

    Returns:
        total_loss : scalar        weighted sum for backward
        pred       : [E_new, 1]    predictions (for logger metrics)

    Side effects:
        Attaches individual loss scalars to batch for logging:
          batch.loss_data  : scalar  (detached)
          batch.loss_eq    : scalar  (detached)
          batch.loss_pde   : scalar  (detached)
    """
    true = batch.y  # [E_new, 1]

    # ══ L_Data: data-fitting loss (normalized space) ══════════════════
    loss_data = F.mse_loss(pred, true)

    # ══ Physics losses (require Phase 3 diffusion outputs) ════════════
    rho_v_final = getattr(batch, 'rho_v_final', None)
    rho_v_history = getattr(batch, 'rho_v_history', None)

    if rho_v_final is not None:
        flow_std = cfg.dataset.flow_std  # σ (scalar, veh/hr)

        # ── L_Eq: terminal equilibrium MSE ────────────────────────────
        # ρ^(K) should → 0 for perfect flow conservation
        rho_final_scaled = rho_v_final / flow_std        # [N, 1], dimensionless
        loss_eq = F.mse_loss(
            rho_final_scaled,
            torch.zeros_like(rho_final_scaled),
        )

        # ── L_PDE: intermediate smoothing L1 ─────────────────────────
        # Average L1‖ρ^(k)/σ‖ over k = 1 … K-1
        #   k=0  skipped: initial shockwave (large by definition)
        #   k=K  skipped: already penalized by L_Eq (stricter MSE)
        #
        # rho_v_history layout: [ρ^(0), ρ^(1), …, ρ^(K)]
        #   len = K+1,  intermediate range = indices 1 … K-1
        if rho_v_history is not None and len(rho_v_history) > 2:
            intermediate_losses = []
            for k in range(1, len(rho_v_history) - 1):
                rho_k_scaled = rho_v_history[k] / flow_std   # [N, 1]
                intermediate_losses.append(
                    F.l1_loss(rho_k_scaled, torch.zeros_like(rho_k_scaled))
                )
            # Mean across K-1 intermediate steps → single scalar
            loss_pde = torch.stack(intermediate_losses).mean()
        else:
            loss_pde = torch.tensor(0.0, device=pred.device)

        # ── Total weighted loss ───────────────────────────────────────
        lambda_eq = cfg.model.lambda_eq      # default 1.0
        lambda_pde = cfg.model.lambda_pde    # default 0.05
        total_loss = (
            loss_data
            + lambda_eq * loss_eq
            + lambda_pde * loss_pde
        )
    else:
        # Baseline model without diffusion loop → data loss only
        loss_eq = torch.tensor(0.0, device=pred.device)
        loss_pde = torch.tensor(0.0, device=pred.device)
        total_loss = loss_data

    # ══ Attach individual losses to batch for logging ═════════════════
    batch.loss_data = loss_data.detach()
    batch.loss_eq = loss_eq.detach()
    batch.loss_pde = loss_pde.detach()

    return total_loss, pred
