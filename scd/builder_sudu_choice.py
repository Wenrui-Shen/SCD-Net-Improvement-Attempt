import torch
import torch.nn as nn

from .scd_encoder_sudu_choice import PretrainingEncoder


# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


class SCD_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        super(SCD_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # queues
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("i_queue", torch.randn(dim, K))
        self.i_queue = nn.functional.normalize(self.i_queue, dim=0)
        self.register_buffer("i_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.enqueue_enabled = True

        # cache for phys settings
        self._phys_cfg = {}
        self._phys_strength = {}

    # === pass-through setters ===
    def set_phys_cfg(self, cfg: dict):
        self._phys_cfg.update(cfg)
        if hasattr(self.encoder_q, 'set_phys_cfg'):
            self.encoder_q.set_phys_cfg(cfg)
        if hasattr(self.encoder_k, 'set_phys_cfg'):
            self.encoder_k.set_phys_cfg(cfg)

    def set_phys_strength(self, beta_s: float, beta_t: float,
                          lambda_phys: float, alpha_gate: float, gamma_gate: float):
        payload = dict(beta_s=beta_s, beta_t=beta_t,
                       lambda_phys=lambda_phys, alpha_gate=alpha_gate, gamma_gate=gamma_gate)
        self._phys_strength.update(payload)
        if hasattr(self.encoder_q, 'set_phys_strength'):
            self.encoder_q.set_phys_strength(**payload)
        if hasattr(self.encoder_k, 'set_phys_strength'):
            self.encoder_k.set_phys_strength(**payload)

    # ★ 新增：选择只在 Q 或 K 侧应用残差/门控
    def set_phys_side(self, residual_side: str = 'both', gate_side: str = 'both'):
        # residual
        res_q = residual_side in ('both', 'q')
        res_k = residual_side in ('both', 'k')
        # gate
        gat_q = gate_side in ('both', 'q')
        gat_k = gate_side in ('both', 'k')

        if hasattr(self.encoder_q, 'set_phys_apply'):
            self.encoder_q.set_phys_apply(residual=res_q, gate=gat_q)
        if hasattr(self.encoder_k, 'set_phys_apply'):
            self.encoder_k.set_phys_apply(residual=res_k, gate=gat_k)

    def set_enqueue_enabled(self, enabled: bool = True):
        self.enqueue_enabled = bool(enabled)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, i_keys):
        N, C = t_keys.shape
        assert self.K % N == 0, f"K={self.K} must be divisible by N={N}"

        t_ptr = int(self.t_queue_ptr)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        self.t_queue_ptr[0] = (t_ptr + N) % self.K

        s_ptr = int(self.s_queue_ptr)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        self.s_queue_ptr[0] = (s_ptr + N) % self.K

        i_ptr = int(self.i_queue_ptr)
        self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
        self.i_queue_ptr[0] = (i_ptr + N) % self.K

    def forward(self, q_input, k_input):
        # queries
        qt, qs, qi = self.encoder_q(q_input)
        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qi = nn.functional.normalize(qi, dim=1)

        # keys
        with torch.no_grad():
            self._momentum_update_key_encoder()
            kt, ks, ki = self.encoder_k(k_input)
            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ki = nn.functional.normalize(ki, dim=1)

        # dot products
        l_pos_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
        l_pos_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
        l_pos_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
        l_pos_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)

        l_neg_ti = torch.einsum('nc,ck->nk', [qt, self.i_queue.clone().detach()])
        l_neg_si = torch.einsum('nc,ck->nk', [qs, self.i_queue.clone().detach()])
        l_neg_it = torch.einsum('nc,ck->nk', [qi, self.t_queue.clone().detach()])
        l_neg_is = torch.einsum('nc,ck->nk', [qi, self.s_queue.clone().detach()])

        logits_ti = torch.cat([l_pos_ti, l_neg_ti], dim=1) / self.T
        logits_si = torch.cat([l_pos_si, l_neg_si], dim=1) / self.T
        logits_it = torch.cat([l_pos_it, l_neg_it], dim=1) / self.T
        logits_is = torch.cat([l_pos_is, l_neg_is], dim=1) / self.T

        labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long, device=logits_ti.device)
        labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long, device=logits_si.device)
        labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long, device=logits_it.device)
        labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long, device=logits_is.device)

        if self.enqueue_enabled:
            self._dequeue_and_enqueue(kt, ks, ki)

        return logits_ti, logits_si, logits_it, logits_is, \
               labels_ti, labels_si, labels_it, labels_is
