import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange

from model.ctrgcn import Model


# ===== 默认参数（你想改就在这里改）=====
KIN_DEFAULTS = dict(
    use_residual=True,   # 残差改语义：Y <- Y + beta * Linear(kin_stats)
    use_gate=True,       # 速度门控：tokens *= (1 + alpha * s_hat)
    beta_t=0.25,          # 时间路残差强度
    beta_s=0.3,          # 空间路残差强度
    alpha_gate=0.25,      # 门控强度
)
M_STREAMS = 2            # 你的 rearrange 里用到了 M=2


def _minmax_norm(x, dim, eps=1e-6):
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer, kin_cfg=None) -> None:
        super().__init__()
        self.d_model = hidden_size

        # ===== GCN backbones =====
        hidden_size = 64
        self.gcn_t = Model(hidden_size)
        self.gcn_s = Model(hidden_size)

        # ===== token embeddings =====
        self.channel_t = nn.Sequential(
            nn.Linear(50 * hidden_size, self.d_model),
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

        # ===== Transformers =====
        encoder_layer = TransformerEncoderLayer(self.d_model, num_head, self.d_model, batch_first=True)
        self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
        self.s_encoder = TransformerEncoder(encoder_layer, num_layer)

        # ===== Kinematics config =====
        cfg = dict(KIN_DEFAULTS)  # 默认启用
        if kin_cfg is not None:
            cfg.update(kin_cfg)
        self.use_residual = bool(cfg.get('use_residual', True))
        self.use_gate     = bool(cfg.get('use_gate', True))
        self.beta_t       = float(cfg.get('beta_t', 0.3))
        self.beta_s       = float(cfg.get('beta_s', 0.3))
        self.alpha_gate   = float(cfg.get('alpha_gate', 0.3))

        # residual heads
        self.kin_proj_t = nn.Sequential(nn.Linear(4, self.d_model), nn.LayerNorm(self.d_model))
        self.kin_proj_s = nn.Sequential(nn.Linear(6, self.d_model), nn.LayerNorm(self.d_model))

        self.M = M_STREAMS

        print(f"[Encoder] use_residual={self.use_residual}, use_gate={self.use_gate}, "
              f"beta_t={self.beta_t}, beta_s={self.beta_s}, alpha_gate={self.alpha_gate}")

    def _compute_kinematics(self, x):
    #仅在这里按原版 CTR-GCN 的方式临时压 M 维来计算速度/加速度，
    #不改变传入 GCN 的主分支数据流（仍由 CTR-GCN 自己压 M）。
    #输入 x 可为 (B, C, T, V, M) 或 (B*M, C, T, V)
    #返回 spd, acc 形状为 (B, M, T, V)
        if x.dim() == 5:
            B, C, T, V, M = x.shape
            x4 = x.permute(0, 4, 1, 2, 3).contiguous().view(B * M, C, T, V)  # 与 CTR-GCN 完全一致
            B_out, M_out = B, M
        else:
        # 若上游已压过，这里按原版约定 M=2 复原 B
            BM, C, T, V = x.shape
            M_out = 2
            assert BM % M_out == 0, f"BM={BM} 不能被 M={M_out} 整除"
            B_out = BM // M_out
            x4 = x

        # 时间差分
        v = x4[:, :, 1:, :] - x4[:, :, :-1, :]            # (B*M, C, T-1, V)
        a = v[:, :, 1:, :] - v[:, :, :-1, :]              # (B*M, C, T-2, V)
        # pad 回到 T
        import torch.nn.functional as F
        v = F.pad(v, (0, 0, 1, 0))                        # (B*M, C, T, V)
        a = F.pad(a, (0, 0, 2, 0))                        # (B*M, C, T, V)

        # 通道范数 -> 速/加速度标量
        spd = v.norm(p=2, dim=1)                          # (B*M, T, V)
        acc = a.norm(p=2, dim=1)                          # (B*M, T, V)

        # 还原到 (B, M, T, V) 以便后续按帧/按关节统计
        spd = spd.view(B_out, M_out, T, V)
        acc = acc.view(B_out, M_out, T, V)
        return spd, acc

    def forward(self, x):
        """
        x: (B*M, C, T, V)
        """
        # ===== GCN feature extraction =====
        vt = self.gcn_t(x)  # (B*M, Cg, T, V)
        vs = self.gcn_s(x)  # (B*M, Cg, T, V)

        # ===== tokens before Transformer =====
        vt = rearrange(vt, '(B M) C T V -> B T (M V C)', M=self.M)
        vt = self.channel_t(vt)  # (B, T, d)

        vs = rearrange(vs, '(B M) C T V -> B (M V) (T C)', M=self.M)
        vs = self.channel_s(vs)  # (B, M*V, d)

        # ===== kinematics from augmented input =====
        spd, acc = self._compute_kinematics(x)          # (B, M, T, V)

        # per-frame strength (B,T,1)
        s_frame = spd.mean(dim=(1, 3))                  # (B,T)
        s_frame = _minmax_norm(s_frame, dim=1).unsqueeze(-1)

        # per-joint strength (B,M*V,1)
        s_joint = spd.mean(dim=2)                       # (B,M,V)
        s_joint = _minmax_norm(s_joint.flatten(1), dim=1).view(s_joint.shape[0], -1).unsqueeze(-1)

        # ===== (1) residual injection =====
        if self.use_residual and (self.beta_t != 0.0 or self.beta_s != 0.0):
            # temporal stats per frame: [mean|v|, max|v|, mean|a|, max|a|]
            mu_spd_t  = spd.mean(dim=3)
            max_spd_t = spd.max(dim=3).values
            mu_acc_t  = acc.mean(dim=3)
            max_acc_t = acc.max(dim=3).values
            k_t = torch.stack([
                mu_spd_t.mean(dim=1),
                max_spd_t.mean(dim=1),
                mu_acc_t.mean(dim=1),
                max_acc_t.mean(dim=1),
            ], dim=-1)                                  # (B,T,4)
            vt = vt + self.beta_t * self.kin_proj_t(k_t)

            # spatial stats per joint: [mean|v|, std|v|, max|v|, mean|a|, max|a|, energy]
            mu_spd_j  = spd.mean(dim=2)
            std_spd_j = spd.std(dim=2)
            max_spd_j = spd.max(dim=2).values
            mu_acc_j  = acc.mean(dim=2)
            max_acc_j = acc.max(dim=2).values
            energy_j  = (spd ** 2).mean(dim=2)
            k_s = torch.stack([mu_spd_j, std_spd_j, max_spd_j, mu_acc_j, max_acc_j, energy_j], dim=-1)  # (B,M,V,6)
            k_s = rearrange(k_s, 'B M V D -> B (M V) D')
            vs = vs + self.beta_s * self.kin_proj_s(k_s)

        # ===== (2) speed-based gating =====
        if self.use_gate and self.alpha_gate != 0.0:
            vt = vt * (1.0 + self.alpha_gate * s_frame)
            vs = vs * (1.0 + self.alpha_gate * s_joint)

        # ===== Transformer refinement =====
        vt = self.t_encoder(vt)  # (B, T, d)
        vs = self.s_encoder(vs)  # (B, M*V, d)

        # ===== Global pooling =====
        vt = vt.amax(dim=1)
        vs = vs.amax(dim=1)

        return vt, vs


class PretrainingEncoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer,
                 num_class=60, kin_cfg=None):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size
        self.encoder = Encoder(hidden_size, num_head, num_layer, kin_cfg=kin_cfg)

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

    def forward(self, x):
        vt, vs = self.encoder(x)
        zt = self.t_proj(vt)
        zs = self.s_proj(vs)
        vi = torch.cat([vt, vs], dim=1)
        zi = self.i_proj(vi)
        return zt, zs, zi


class DownstreamEncoder(nn.Module):
    def __init__(self, hidden_size, num_head, num_layer, num_class=60, kin_cfg=None):
        super(DownstreamEncoder, self).__init__()
        self.d_model = hidden_size
        self.encoder = Encoder(hidden_size, num_head, num_layer, kin_cfg=kin_cfg)
        self.fc = nn.Linear(2 * self.d_model, num_class)

    def forward(self, x, knn_eval=False):
        vt, vs = self.encoder(x)
        vi = torch.cat([vt, vs], dim=1)
        if knn_eval:
            return vi
        else:
            return self.fc(vi)
