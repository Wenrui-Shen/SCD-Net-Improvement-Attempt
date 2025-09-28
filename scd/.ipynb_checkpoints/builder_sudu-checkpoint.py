import torch
import torch.nn as nn

from .scd_encoder_sudu import PretrainingEncoder


def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


class SCD_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07, kin_cfg=None):
        super(SCD_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        print(" moco parameters", K, m, T)

        # 不强制传 kin_cfg；编码器内部会用默认参数
        if kin_cfg is None:
            self.encoder_q = PretrainingEncoder(**args_encoder)
            self.encoder_k = PretrainingEncoder(**args_encoder)
        else:
            self.encoder_q = PretrainingEncoder(**args_encoder, kin_cfg=kin_cfg)
            self.encoder_k = PretrainingEncoder(**args_encoder, kin_cfg=kin_cfg)

        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, i_keys):
        N, C = t_keys.shape
        assert self.K % N == 0

        t_ptr = int(self.t_queue_ptr)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K
        self.s_queue_ptr[0] = s_ptr

        i_ptr = int(self.i_queue_ptr)
        self.i_queue[:, i_ptr:i_ptr + N] = i_keys.T
        i_ptr = (i_ptr + N) % self.K
        self.i_queue_ptr[0] = i_ptr

    def forward(self, q_input, k_input):
        qt, qs, qi = self.encoder_q(q_input)
        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qi = nn.functional.normalize(qi, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            kt, ks, ki = self.encoder_k(k_input)
            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            ki = nn.functional.normalize(ki, dim=1)

        l_pos_ti = torch.einsum('nc,nc->n', [qt, ki]).unsqueeze(1)
        l_pos_si = torch.einsum('nc,nc->n', [qs, ki]).unsqueeze(1)
        l_pos_it = torch.einsum('nc,nc->n', [qi, kt]).unsqueeze(1)
        l_pos_is = torch.einsum('nc,nc->n', [qi, ks]).unsqueeze(1)

        l_neg_ti = torch.einsum('nc,ck->nk', [qt, self.i_queue.clone().detach()])
        l_neg_si = torch.einsum('nc,ck->nk', [qs, self.i_queue.clone().detach()])
        l_neg_it = torch.einsum('nc,ck->nk', [qi, self.t_queue.clone().detach()])
        l_neg_is = torch.einsum('nc,ck->nk', [qi, self.s_queue.clone().detach()])

        logits_ti = torch.cat([l_pos_ti, l_neg_ti], dim=1)
        logits_si = torch.cat([l_pos_si, l_neg_si], dim=1)
        logits_it = torch.cat([l_pos_it, l_neg_it], dim=1)
        logits_is = torch.cat([l_pos_is, l_neg_is], dim=1)

        logits_ti /= self.T
        logits_si /= self.T
        logits_it /= self.T
        logits_is /= self.T

        labels_ti = torch.zeros(logits_ti.shape[0], dtype=torch.long).cuda()
        labels_si = torch.zeros(logits_si.shape[0], dtype=torch.long).cuda()
        labels_it = torch.zeros(logits_it.shape[0], dtype=torch.long).cuda()
        labels_is = torch.zeros(logits_is.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(kt, ks, ki)

        return logits_ti, logits_si, logits_it, logits_is, \
               labels_ti, labels_si, labels_it, labels_is
