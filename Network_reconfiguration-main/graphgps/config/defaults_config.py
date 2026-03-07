from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    # ── Phase 4：ST-PINN 三部分物理损失超参数 ──────────────────────────────
    #   L_total = L_Data + λ_eq * L_Eq + λ_pde * L_PDE
    #
    # λ_eq: 终端平衡约束权重
    #   L_Eq = MSE(ρ_v^(K) / σ, 0)
    #   Forces terminal pressure → 0 (flow conservation satisfied).
    cfg.model.lambda_eq = 1.0

    # λ_pde: 中间步平滑约束权重
    #   L_PDE = mean_{k=1}^{K-1} L1(ρ_v^(k) / σ, 0)
    #   Encourages monotonic pressure decay across pseudo-time steps.
    #   Small weight (0.05) acts as regularizer without dominating L_Data.
    cfg.model.lambda_pde = 0.05

    # (Legacy) 旧守恒损失参数，保留用于基线模型向后兼容
    cfg.model.lambda_cons = 0.1
    cfg.model.cons_norm = 'l2'
