import torch

class EMA:
    def __init__(self, model, decay=0.9999, device=None):
        """
        model : nn.Module whose params we track
        decay : float in [0,1), higher -> slower updates (0.999-0.9999 typical)
        device : if provided, store shadow on this device
        """
        self.decay = decay
        self.device = device
        # shadow stores floating tensors (clones of model params)
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device) if device is not None else p.detach().clone()
        self.collected_backup = None

    @torch.no_grad()
    def update(self, model):
        """Call after optimizer.step(). Updates shadow = decay*shadow + (1-decay)*param"""
        for name, p in model.named_parameters():
            if p.requires_grad:
                s = self.shadow[name]
                s.mul_(self.decay).add_(p.detach().to(s.device), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model):
        """Swap current model params with EMA params (for sampling)."""
        self.collected_backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.collected_backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def restore(self, model):
        """Restore original params after sampling."""
        assert self.collected_backup is not None, "No backup found — call apply_shadow first"
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.collected_backup[name].to(p.device))
        self.collected_backup = None
    # ---------------------------
    # Add these for saving/loading
    # ---------------------------
    def state_dict(self):
        # return just the shadow weights
        return {k: v.cpu() for k,v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()
