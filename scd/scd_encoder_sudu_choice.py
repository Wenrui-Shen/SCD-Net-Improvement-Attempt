import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange

from model.ctrgcn import Model


def _quantile_norm(x, q_low=0.10, q_high=0.90, eps=1e-6):
    x_flat = x.reshape(-1)
    if x_flat.numel() == 0:
        return x
    low = torch.quantile(x_flat, q_low)
    high = torch.quantile(x_flat, q_high)
    den = torch.clamp(high - low, min=eps)
    out = (x - low) / den
    return out.clamp_(0., 1.)


def _smooth_time(x, k=3):
    # x: (B, M, C, T, V)
    if k <= 1:
        return x
    pad = (k - 1) // 2
    x_pad = F.pad(x, (0, 0, pad, pad), mode='replicate')
    B, M, C, T, V = x.shape
    kernel = torch.ones((1, 1, k, 1), device=x.device, dtype=x.dtype) / k
    x_ = x_pad.permute(0, 1, 4, 2, 3).reshape(B*M*V, C, 1, T + 2*pad)
    x_f = F.conv2d(x_, kernel)
    x_s = x_f.reshape(B, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
    return x_s


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer) -> None:
        super().__init__()
        self.d_model = hidden_size

        hidden_size = 64
        self.gcn_t = Model(hidden_size)
        self.gcn_s = Model(hidden_size)

        self.channel_t = nn.Sequential(
            nn.Linear(50*hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        self.channel_s = nn.Sequential(
            nn.Linear(64 * hidden_size, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, self.d_model),
        )

        encoder_layer = TransformerEncoderLayer(self.d_model, num_head, self.d_model, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)

        # ===== 写死为默认开启；仍支持被 builder 覆盖 =====
        self.phys_cfg = dict(
            use_kin_residual=True,
            use_phys_bias=False,   # 占位：未改 logits
            use_kv_gate=True,
            phys_use_acc=True,
            phys_mix_a=0.3,
            attn_bias_rownorm=False
        )
        self.phys_strength = dict(
            beta_s=0.30, beta_t=0.30,
            lambda_phys=0.0,
            alpha_gate=0.25, gamma_gate=0.25
        )
        # 哪个分支应用（默认都应用；builder 会按 Q/K 改写）
        self.phys_apply = dict(residual=True, gate=True)

        # 残差投影头
        dks, dkt = 10, 6
        self.kin_proj_s = nn.Sequential(
            nn.Linear(dks, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        self.kin_proj_t = nn.Sequential(
            nn.Linear(dkt, self.d_model),
            nn.LayerNorm(self.d_model)
        )

    # 仍保留这些 setter，便于 builder 写死后下发
    def set_phys_cfg(self, cfg: dict):
        self.phys_cfg.update(cfg)

    def set_phys_strength(self, beta_s: float, beta_t: float,
                          lambda_phys: float, alpha_gate: float, gamma_gate: float):
        self.phys_strength.update(dict(
            beta_s=beta_s, beta_t=beta_t,
            lambda_phys=lambda_phys,
            alpha_gate=alpha_gate, gamma_gate=gamma_gate
        ))

    def set_phys_apply(self, residual: bool = None, gate: bool = None):
        if residual is not None:
            self.phys_apply['residual'] = bool(residual)
        if gate is not None:
            self.phys_apply['gate'] = bool(gate)

    @torch.no_grad()
    def _compute_kinematics(self, x, M=2, smooth=True):
        BM, C, T, V = x.shape
        assert BM % M == 0, "BM must be divisible by M=2"
        B = BM // M
        x5 = x.view(B, M, C, T, V)
        xyz = x5[:, :, :3, :, :]
        if smooth:
            xyz = _smooth_time(xyz, k=3)

        v = xyz[:, :, :, 1:, :] - xyz[:, :, :, :-1, :]
        a = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]

        v = F.pad(v, (0, 0, 1, 0))
        a = F.pad(a, (0, 0, 2, 0))

        spd = torch.norm(v, p=2, dim=2)   # (B,M,T,V)
        acc = torch.norm(a, p=2, dim=2)   # (B,M,T,V)

        mix_a = self.phys_cfg.get('phys_mix_a', 0.3) if self.phys_cfg.get('phys_use_acc', False) else 0.0
        s_joint_time = spd + mix_a * acc
        s_frame_joint = spd + mix_a * acc

        s_joint = s_joint_time.mean(dim=2).reshape(B, M*V, 1)
        s_joint = _quantile_norm(s_joint)

        s_frame = s_frame_joint.mean(dim=3).mean(dim=1).unsqueeze(-1)  # (B,T,1)
        s_frame = _quantile_norm(s_frame)

        mu_spd  = spd.mean(dim=2); std_spd = spd.std(dim=2); max_spd = spd.max(dim=2).values
        mu_acc  = acc.mean(dim=2); std_acc = acc.std(dim=2); max_acc = acc.max(dim=2).values

        v_dir = F.normalize(v + 1e-6, dim=2)
        dir_mean = v_dir.mean(dim=3).permute(0,1,3,2).reshape(B, M*V, 3)

        energy = (spd ** 2).mean(dim=2)

        k_s = torch.cat([
            mu_spd.reshape(B, M*V, 1),
            std_spd.reshape(B, M*V, 1),
            max_spd.reshape(B, M*V, 1),
            mu_acc.reshape(B, M*V, 1),
            std_acc.reshape(B, M*V, 1),
            max_acc.reshape(B, M*V, 1),
            dir_mean,
            energy.reshape(B, M*V, 1)
        ], dim=-1)  # (B,MV,10)

        mu_spd_t  = spd.mean(dim=3).mean(dim=1, keepdim=True).transpose(2,3).squeeze(1)
        max_spd_t = spd.max(dim=3).values.max(dim=1).values
        mu_acc_t  = acc.mean(dim=3).mean(dim=1, keepdim=True).transpose(2,3).squeeze(1)
        max_acc_t = acc.max(dim=3).values.max(dim=1).values

        p = spd / (spd.sum(dim=3, keepdim=True) + 1e-6)
        entropy = (-p * (p + 1e-6).log()).sum(dim=3).mean(dim=1)
        entropy = entropy / (torch.log(torch.tensor(V, device=x.device, dtype=x.dtype)) + 1e-6)

        v_dir_t = v_dir.mean(dim=1)
        cos_phase = (v_dir_t[:, :, 1:, :] * v_dir_t[:, :, :-1, :]).sum(dim=1).mean(dim=2)
        cos_phase = F.pad(cos_phase, (0, 1))

        k_t = torch.stack([mu_spd_t, max_spd_t, mu_acc_t, max_acc_t, entropy, cos_phase], dim=-1)  # (B,T,6)
        return s_joint, s_frame, k_s, k_t

    def forward(self, x):
        # 原始管线
        vt = self.gcn_t(x)
        vt = rearrange(vt, '(B M) C T V -> B T (M V C)', M=2)
        vt = self.channel_t(vt)    # (B, T, d_model)

        vs = self.gcn_s(x)
        vs = rearrange(vs, '(B M) C T V -> B (M V) (T C)', M=2)
        vs = self.channel_s(vs)    # (B, M*V, d_model)

        # 物理先验：残差 + 门控（按侧生效）
        use_res = self.phys_cfg.get('use_kin_residual', False) and self.phys_apply.get('residual', True)
        use_gate = self.phys_cfg.get('use_kv_gate', False) and self.phys_apply.get('gate', True)
        beta_s = float(self.phys_strength.get('beta_s', 0.0))
        beta_t = float(self.phys_strength.get('beta_t', 0.0))
        alpha_gate = float(self.phys_strength.get('alpha_gate', 0.0))

        if (use_res and (beta_s > 0 or beta_t > 0)) or (use_gate and alpha_gate > 0):
            with torch.no_grad():
                s_joint, s_frame, k_s, k_t = self._compute_kinematics(x, M=2, smooth=True)

            if use_res and beta_s > 0:
                vs = vs + beta_s * self.kin_proj_s(k_s)
            if use_res and beta_t > 0:
                vt = vt + beta_t * self.kin_proj_t(k_t)

            if use_gate and alpha_gate > 0:
                vt = vt * (1.0 + alpha_gate * s_frame)  # (B,T,1) → 广播到 C
                vs = vs * (1.0 + alpha_gate * s_joint)  # (B,MV,1) → 广播到 C

        # Transformer 精炼
        vt = self.t_encoder(vt)
        vs = self.s_encoder(vs)

        # 池化
        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)

        return vt, vs


class PretrainingEncoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        self.t_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.s_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        self.i_proj = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    # 透传（builder 会调用；若你不需要，也可以不调用）
    def set_phys_cfg(self, cfg: dict):
        if hasattr(self.encoder, 'set_phys_cfg'):
            self.encoder.set_phys_cfg(cfg)

    def set_phys_strength(self, beta_s: float, beta_t: float,
                          lambda_phys: float, alpha_gate: float, gamma_gate: float):
        if hasattr(self.encoder, 'set_phys_strength'):
            self.encoder.set_phys_strength(beta_s, beta_t, lambda_phys, alpha_gate, gamma_gate)

    def set_phys_apply(self, residual: bool = None, gate: bool = None):
        if hasattr(self.encoder, 'set_phys_apply'):
            self.encoder.set_phys_apply(residual=residual, gate=gate)

    def forward(self, x):
        vt, vs = self.encoder(x)
        zt = self.t_proj(vt)
        zs = self.s_proj(vs)
        vi = torch.cat([vt, vs], dim=1)
        zi = self.i_proj(vi)
        return zt, zs, zi


class DownstreamEncoder(nn.Module):
    """hierarchical encoder network + classifier"""

    def __init__(self, 
                 hidden_size, num_head, num_layer,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.encoder = Encoder(
            hidden_size, num_head, num_layer,
        )

        self.fc = nn.Linear(2 * self.d_model, num_class)

    # 可选：下游也能用同样的 setter
    def set_phys_cfg(self, cfg: dict):
        if hasattr(self.encoder, 'set_phys_cfg'):
            self.encoder.set_phys_cfg(cfg)

    def set_phys_strength(self, beta_s: float, beta_t: float,
                          lambda_phys: float, alpha_gate: float, gamma_gate: float):
        if hasattr(self.encoder, 'set_phys_strength'):
            self.encoder.set_phys_strength(beta_s, beta_t, lambda_phys, alpha_gate, gamma_gate)

    def set_phys_apply(self, residual: bool = None, gate: bool = None):
        if hasattr(self.encoder, 'set_phys_apply'):
            self.encoder.set_phys_apply(residual=residual, gate=gate)

    def forward(self, x, knn_eval=False):
        vt, vs = self.encoder(x)
        vi = torch.cat([vt, vs], dim=1)
        if knn_eval:
            return vi
        else:
            return self.fc(vi)
