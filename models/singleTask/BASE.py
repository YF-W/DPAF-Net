import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_views=8):
        super().__init__()

        self.num_views = num_views

        self.view_linears = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
            for _ in range(num_views)
        ])

        self.view_score = nn.Linear(hidden_dim, 1)

        self.view_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_views)
        ])

    def forward(self, x):

        views = []

        for linear, norm in zip(self.view_linears, self.view_norms):
            v = linear(x)
            v = torch.relu(v)
            v = norm(v)
            views.append(v)

        Z = torch.stack(views, dim=1)

        scores = self.view_score(Z).squeeze(-1)

        weights = torch.softmax(scores, dim=-1)

        threshold = 1.0 / self.num_views

        temperature = 20

        mask = torch.sigmoid(
            temperature * (weights - threshold)
        )

        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        Z = Z * weights.unsqueeze(-1)

        return Z, weights

class ViewStructureEncoder(nn.Module):

    def __init__(self, dim, num_heads=4, num_layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, Z):

        H = self.encoder(Z)

        H = self.norm(H + Z)

        return H

class PersonalityGenerator(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

        self.pool_query = nn.Parameter(
            torch.randn(1, 1, dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
)

    def forward(self, H):

        B = H.size(0)

        query = self.pool_query.expand(B, -1, -1)

        pooled, attn_map = self.attn(
            query,
            H,
            H
        )

        p = pooled.squeeze(1)

        p_res = p

        p = self.mlp(p)

        p = self.norm(p + p_res)

        return p

class PersonalityRouting(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.score_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, H, p):

        B,K,D = H.shape

        hi = H.unsqueeze(2).expand(B,K,K,D)
        hj = H.unsqueeze(1).expand(B,K,K,D)

        p_expand = p.unsqueeze(1).unsqueeze(2).expand(B,K,K,D)

        pair = torch.cat([hi,hj,p_expand], dim=-1)

        scores = self.score_mlp(pair).squeeze(-1)

        alpha = torch.softmax(scores, dim=-1)

        Hr = torch.matmul(alpha, H)

        Hr = self.dropout(Hr)

        Hr = self.norm(Hr + H)

        return Hr, alpha

class StructureRefiner(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        self.dropout = nn.Dropout(0.1)

        self.norm = nn.LayerNorm(dim)

    def forward(self, Hr, alpha, p):

        B,K,D = Hr.shape

        hi = Hr.unsqueeze(2).expand(B,K,K,D)
        hj = Hr.unsqueeze(1).expand(B,K,K,D)
        p_expand = p.unsqueeze(1).unsqueeze(2).expand(
            B, K, K, D
)

        edge = torch.cat([
            hi,
            hj,
            p_expand,
            alpha.unsqueeze(-1)
        ], dim=-1)

        delta = 0.1 * torch.tanh(
            self.edge_mlp(edge).squeeze(-1)
        )

        A_star = torch.softmax(alpha + delta, dim=-1)

        H_star = torch.matmul(A_star, Hr)

        H_star = self.dropout(H_star)

        H_star = Hr + 0.3 * H_star

        H_star = self.norm(H_star)

        return H_star, A_star

class PersonalityInteraction(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.personality_control = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.gamma = nn.Linear(dim, dim)

        self.beta = nn.Linear(dim, dim)

        self.policy = nn.Linear(dim, 4)

        self.dropout = nn.Dropout(0.1)

        self.p_norm = nn.LayerNorm(dim)

        self.z_norm = nn.LayerNorm(dim)

        self.dominant_proj = nn.Linear(dim, dim)

        self.reinterpret_proj = nn.Linear(dim, dim)

    def forward(self, H_star):

        z = H_star.mean(dim=1)

        p_prime = self.personality_control(z)

        p_prime = self.p_norm(p_prime + z)
        p_prime = torch.tanh(p_prime)

        consistency_score = torch.matmul(
            H_star,
            p_prime.unsqueeze(-1)
        ).squeeze(-1)

        consistency_weight = torch.softmax(
            consistency_score,
            dim=-1
        )

        O1 = torch.sum(
            H_star * consistency_weight.unsqueeze(-1),
            dim=1
        )

        hi = H_star.unsqueeze(2)
        hj = H_star.unsqueeze(1)

        hi_norm = F.normalize(
            hi,
            dim=-1
        )

        hj_norm = F.normalize(
            hj,
            dim=-1
        )

        conflict = -torch.sum(
            hi_norm * hj_norm,
            dim=-1
        )

        conflict_map = torch.softmax(
            conflict / 0.3,
            dim=-1
        )

        conflict_feat = torch.matmul(
            conflict_map,
            H_star
        )

        O2 = conflict_feat.mean(dim=1)

        hd = self.dominant_proj(H_star)

        dom_score = torch.matmul(
            hd,
            p_prime.unsqueeze(-1)
        ).squeeze(-1)

        dom_weight = torch.softmax(
            dom_score / 0.3,
            dim=-1
        )

        A3 = (
            dom_weight.unsqueeze(-1)
            *
            dom_weight.unsqueeze(1)
        )

        O3 = torch.matmul(
            A3,
            H_star
        ).mean(dim=1)

        hr = self.reinterpret_proj(H_star)

        gamma = torch.tanh(
            self.gamma(p_prime)
        )

        beta = self.beta(p_prime)

        reinterpret = (
            gamma.unsqueeze(1) * hr
            + beta.unsqueeze(1)
        )

        A4 = torch.matmul(
            reinterpret,
            reinterpret.transpose(-1, -2)
        )

        A4 = torch.softmax(
            A4,
            dim=-1
        )

        O4 = torch.matmul(
            A4,
            reinterpret
        ).mean(dim=1)

        pi_logits = self.policy(p_prime)

        pi = torch.softmax(
            pi_logits / 0.5,
            dim=-1
        )

        Zp = (
            pi[:, 0:1] * O1 +
            pi[:, 1:2] * O2 +
            pi[:, 2:3] * O3 +
            pi[:, 3:4] * O4
        )

        Zp = self.dropout(Zp)

        Zp = self.z_norm(Zp + z)

        return Zp, pi, p_prime

class PersonalityDecision(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.confidence = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        self.threshold = nn.Linear(dim, 1)

        self.head = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        self.personality_confidence = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, Zp, p_prime, modal_feats):

        B, M, D = modal_feats.shape

        p_expand = p_prime.unsqueeze(1).expand(B, M, D)
        z_expand = Zp.unsqueeze(1).expand(B, M, D)

        gate_input = torch.cat(
            [
                modal_feats,
                p_expand,
                z_expand
            ],
            dim=-1
        )

        gate_bias = self.gate_mlp(
            gate_input
        ).squeeze(-1)

        confidence_input = torch.cat(
            [modal_feats, p_expand],
            dim=-1
        )

        scores = self.confidence(
            confidence_input
        ).squeeze(-1)

        tau = torch.sigmoid(
            self.threshold(p_prime)
        )

        gates = torch.sigmoid(
            scores
            +
            gate_bias
            -
            tau
        )

        gates = gates / (
            gates.sum(dim=-1, keepdim=True)
            + 1e-8
        )

        fused_modal = torch.sum(
            modal_feats * gates.unsqueeze(-1),
            dim=1
        )

        fused_modal = self.dropout(fused_modal)

        confidence = torch.sigmoid(
            self.personality_confidence(p_prime)
        )

        final_feat = self.norm(
            fused_modal
            +
            confidence * Zp
        )
        out = self.head(final_feat)

        return out, gates

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()

        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device('cpu')

        self.text_in, self.audio_in, self.video_in = args.feature_dims[:3]
        self.dim = getattr(args, 'hidden_dim', 128)

        self.view_projector = MultiViewProjector(self.dim, self.dim, num_views=8)

        self.view_encoder = ViewStructureEncoder(self.dim)

        self.personality_generator = PersonalityGenerator(self.dim)

        self.routing = PersonalityRouting(self.dim)

        self.refiner = StructureRefiner(self.dim)

        self.interaction = PersonalityInteraction(self.dim)

        self.decision = PersonalityDecision(self.dim)

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_in, self.dim),
            nn.ReLU(),
            nn.LayerNorm(self.dim),
            nn.Dropout(0.1)
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_in, self.dim),
            nn.ReLU(),
            nn.LayerNorm(self.dim),
            nn.Dropout(0.1)
        )

        self.video_encoder = nn.Sequential(
            nn.Linear(self.video_in, self.dim),
            nn.ReLU(),
            nn.LayerNorm(self.dim),
            nn.Dropout(0.1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.ReLU(),
            nn.LayerNorm(self.dim),
            nn.Dropout(0.1)
        )

    def _safe_pool(self, x):

        if x.dim() == 3:
            return x.mean(dim=1)
        return x

    def forward(self, text, audio, video):

        t = self._safe_pool(text)
        a = self._safe_pool(audio)
        v = self._safe_pool(video)

        t_feat = self.text_encoder(t)
        a_feat = self.audio_encoder(a)
        v_feat = self.video_encoder(v)

        fused = torch.cat([t_feat, a_feat, v_feat], dim=-1)
        z = self.fusion(fused)

        Z, weights = self.view_projector(z)

        if self.training and torch.rand(1).item() < 0.01:
            active_views = (weights > 1e-2).float().sum(dim=-1)
            print("Active views (avg):", active_views.mean().item())

        if self.training and torch.rand(1).item() < 0.01:
            print("Weight stats:",
                "max =", weights.max().item(),
                "min =", weights.min().item(),
                "entropy =", (-weights * torch.log(weights + 1e-8)).sum(dim=-1).mean().item())

        H = self.view_encoder(Z)

        p = self.personality_generator(H)

        Hr, alpha = self.routing(H, p)

        H_star, A_star = self.refiner(Hr, alpha, p)

        Zp, pi, p_prime = self.interaction(H_star)

        modal_feats = torch.stack([
            t_feat,
            a_feat,
            v_feat
        ], dim=1)

        out, gates = self.decision(
            Zp,
            p_prime,
            modal_feats
        )

        return {
            'result': out,

            'H': H,
            'Hr': Hr,
            'H_star': H_star,

            'p': p,

            'alpha': alpha,
            'A_star': A_star,

            'pi': pi,
            'gates': gates,

            'view_weights': weights,
            'p_prime': p_prime,

            'Feature_t': t_feat,
            'Feature_a': a_feat,
            'Feature_v': v_feat,
            'Feature_f': z,

            'views': Z,
        }
