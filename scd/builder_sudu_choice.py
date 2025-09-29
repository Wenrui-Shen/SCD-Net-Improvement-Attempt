import torch
import torch.nn as nn

from .scd_encoder import PretrainingEncoder


# ================== 写死的全局配置（你可以按需修改） ==================
PHYS_CFG = dict(
    USE_KIN_RESIDUAL=True,   # 开启残差注入：Y <- Y + beta * W(kin_stats)
    USE_KV_GATE=True,        # 开启速度门控：token * (1 + alpha * s_hat)
    USE_PHYS_BIAS=False,     # 占位（本实现未改 logits）
    USE_ACC=True,            # 速度打分混入加速度
    MIX_A=0.3,               # s = |v| + MIX_A * |a|
)

PHYS_STRENGTH = dict(
    BETA_S=0.30,             # 空间路残差强度
    BETA_T=0.30,             # 时间路残差强度
    LAMBDA_PHYS=0.0,         # 占位（未用）
    ALPHA_GATE=0.25,         # 门控强度
    GAMMA_GATE=None          # 占位（未用；等同 ALPHA_GATE）
)

# 只在哪一侧应用（'q' / 'k' / 'both' / 'none'）
APPLY_SIDE_RESIDUAL = 'q'
APPLY_SIDE_GATE     = 'q'
# ===============================================================


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
        """
        args_encoder: encoder args
        dim: feature dim pushed to MoCo queues
        K: queue size
        m: momentum for key encoder update
        T: softmax temperature
        """
        super(SCD_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        # 两个分支
        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # ==== 把“写死”的配置与强度下发到 Q/K 两侧 ====
        # 配置
        cfg_payload = dict(
            use_kin_residual=PHYS_CFG['USE_KIN_RESIDUAL'],
            use_phys_bias=PHYS_CFG['USE_PHYS_BIAS'],
            use_kv_gate=PHYS_CFG['USE_KV_GATE'],
            phys_use_acc=PHYS_CFG['USE_ACC'],
            phys_mix_a=PHYS_CFG['MIX_A'],
            attn_bias_rownorm=False  # 占位
        )
        for enc in [self.encoder_q, self.encoder_k]:
            if hasattr(enc, 'set_phys_cfg'):
                enc.set_phys_cfg(cfg_payload)

        # 强度
        gamma_gate = PHYS_STRENGTH['ALPHA_GATE'] if PHYS_STRENGTH['GAMMA_GATE'] is None else PHYS_STRENGTH['GAMMA_GATE']
        strength_payload = dict(
            beta_s=PHYS_STRENGTH['BETA_S'],
            beta_t=PHYS_STRENGTH['BETA_T'],
            lambda_phys=PHYS_STRENGTH['LAMBDA_PHYS'],
            alpha_gate=PHYS_STRENGTH['ALPHA_GATE'],
            gamma_gate=gamma_gate
        )
        for enc in [self.encoder_q, self.encoder_k]:
            if hasattr(enc, 'set_phys_strength'):
                enc.set_phys_strength(**strength_payload)

        # 只在 Q 或 K 侧启用（写死）
        def _parse_side(side):
            return dict(
                q = (side in ('q','both')),
                k = (side in ('k','both'))
            )
        res_side = _parse_side(APPLY_SIDE_RESIDUAL)
        gat_side = _parse_side(APPLY_SIDE_GATE)

        if hasattr(self.encoder_q, 'set_phys_apply'):
            self.encoder_q.set_phys_apply(residual=res_side['q'], gate=gat_side['q'])
        if hasattr(self.encoder_k, 'set_phys_apply'):
            self.encoder_k.set_phys_apply(residual=res_side['k'], gate=gat_side['k'])
        # ================================================

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

    def set_enqueue_enabled(self, enabled: bool = True):
        self.enqueue_enabled = bool(enabled)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
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

        # interactive dot products
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
