""" Cosine Similarity between Object Query and Distillation Query """


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.swin_detr import SwinDETR
from eval import get_args_parser  # args def function


@torch.no_grad()
def compute_layerwise_query_cosine(model, features, mask, pos):
    """
    Args:
        model: SwinDETR (eval)
        features: [B, C, H, W]
        mask:     [B, H, W]
        pos:      [HW, B, C]
    Returns:
        torch.Tensor [num_decoder_layers] — cosine similarity per layer
    """
    # transformer_encoder_decoder 의 intermediate 출력: [L, B, Q+DQ, D]
    out = model.transformer_encoder_decoder(
        features,                          # [B,C,H,W]
        mask,                              # [B,H,W]
        model.query_embed.weight,          # [Q, D]
        model.distill_query_embed.weight,  # [DQ, D]
        pos                                 # [HW, B, D]
    )
    # out: [L, B, Q+DQ, D]
    L, B, Qtot, D = out.shape
    Q, DQ = model.num_query, model.num_distill_query

    cos_sims = []
    for l in range(L):
        layer_feat  = out[l]           # [B, Q+DQ, D]
        obj_feat    = layer_feat[:, :Q, :]     # [B, Q, D]
        dist_feat   = layer_feat[:, Q:, :]     # [B, DQ, D]
        # batch & query dims 평균
        obj_mean  = obj_feat.mean(dim=(0, 1))   # [D]
        dist_mean = dist_feat.mean(dim=(0, 1))  # [D]
        # cosine similarity
        cos = F.cosine_similarity(
            obj_mean.unsqueeze(0), 
            dist_mean.unsqueeze(0), 
            dim=-1
        )  # [1]
        cos_sims.append(cos.item())

    return torch.tensor(cos_sims)  # [L]


def extract_features_mask_pos(model, images):
    """
    SwinDETR에서 backbone + reduction + position embedding 과정을 거쳐
    features, mask, pos 를 뽑아서 반환합니다.
    """
    # 1) backbone hook 으로 Swin Transformer norm 이후 feature 획득
    forward_features = {}
    hook = model.swin_transformer_backbone.norm.register_forward_hook(
        lambda module, inp, out: forward_features.setdefault('feat', out)
    )
    _ = model.swin_transformer_backbone(images)  # [B, L, C]
    hook.remove()

    # 2) dimension reduction: [B, L, C] -> [B, C_red, H, W]
    feat = forward_features['feat']               # [B, L, C_backbone]
    reduced = model.reduction(feat)               # [B, L, hidden_dim]
    B, L, D = reduced.shape
    H = W = int(L ** 0.5)
    features = (reduced
                .permute(0, 2, 1)                 # [B, D, L]
                .reshape(B, D, H, W))             # [B, D, H, W]

    # 3) positional encoding
    pos, mask = model.position_embedding(features)
    return features, mask, pos


if __name__ == "__main__":
    # --- 0) args & device 세팅 ---
    parser = get_args_parser()
    args = parser.parse_args()  # 예: --weights_path, --device 등
    device = torch.device(args.device)

    # --- 1) 모델 로드 ---
    model = SwinDETR(args).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # --- 2) 이미지 배치 준비 (예시: 단일 이미지) ---
    #    실제 평가용 배치 대신 여기에 고유 이미지 폴더/리스트를 넣으시면 됩니다.
    preprocess = transforms.Compose([
        transforms.Resize((args.target_height, args.target_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open("test.jpg").convert("RGB")
    images = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]
    # images = torch.randn(16, 3, 224, 224)

    # --- 3) features, mask, pos 추출 ---
    features, mask, pos = extract_features_mask_pos(model, images)

    # --- 4) 레이어별 cosine similarity 계산 ---
    cosine_per_layer = compute_layerwise_query_cosine(model, features, mask, pos)

    # --- 5) 결과 출력 ---
    print("Layer-wise Object‑Query vs Distill‑Query cosine similarity:")
    for layer_idx, cos in enumerate(cosine_per_layer, start=1):
        print(f"  Layer {layer_idx:2d}: {cos:.4f}")
