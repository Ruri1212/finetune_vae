import argparse
import os

import bitsandbytes as bnb
import lpips
import torch
import torch.nn.functional as F
import xformers
from accelerate import Accelerator
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm


# --------------------------------------------------------------
# 学習画像クラス: 画像ファイルのリストを返すデータセット
# --------------------------------------------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, txt_file_path, transform=None):
        """
        txt_file_path: 画像ファイルのパスが列挙されたテキストファイル。
                    空白行が含まれていてもスキップします。
        transform: torchvision.transforms などの前処理
        """
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 空白行や改行だけの行はスキップし、strip() で前後の空白を削除
        self.image_paths = [line.strip() for line in lines if line.strip()]
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# =====================　損失計算に関する函数　=====================
PATCH_SIZE = 64
STRIDE = 32
# 合計のパッチ数 （1024*1024)の画像の場合：　(1024-64)//32 + 1 = 31x31 = 961
# --------------------------------------------------------------
# パッチを取り出す関数
# --------------------------------------------------------------
def extract_patches(image, patch_size=PATCH_SIZE, stride=STRIDE):
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    B, C, patch_H, patch_W, _, _ = patches.shape
    patches = patches.reshape(B, C, patch_H * patch_W, patch_size, patch_size)
    return patches

# --------------------------------------------------------------
# パッチ単位のMSE
# --------------------------------------------------------------
def patch_based_mse_loss(real_images, recon_images, patch_size=PATCH_SIZE, stride=STRIDE):
    real_patches = extract_patches(real_images, patch_size, stride)
    recon_patches = extract_patches(recon_images, patch_size, stride)
    return F.mse_loss(real_patches, recon_patches)

# --------------------------------------------------------------
# パッチ単位のLPIPS
# --------------------------------------------------------------
def patch_based_lpips_loss(lpips_model, real_images, recon_images, patch_size=PATCH_SIZE, stride=STRIDE):
    # bf16 -> float32 へ変換（LPIPS は float32 を想定）
    real_images_f32 = real_images.to(torch.float32)
    recon_images_f32 = recon_images.to(torch.float32)

    real_patches = extract_patches(real_images_f32, patch_size, stride)
    recon_patches = extract_patches(recon_images_f32, patch_size, stride)
    
    B, C, P, H, W = real_patches.shape
    lpips_sum = 0.0
    for i in range(P):
        rp = real_patches[:, :, i, :, :]
        xp = recon_patches[:, :, i, :, :]
        lpips_sum += lpips_model(rp, xp).mean()
    return lpips_sum / P
# ===============================================================




def main(args):
    # 1) Accelerate の初期化
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    print("Accelerator initialized with device:", accelerator.device)

    # 2) データセットとデータローダーの作成
    transform = transforms.Compose([
        v2.Resize(args.resolution, interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomCrop(args.crop_size),
        transforms.ToTensor(),
        v2.Normalize([0.5], [0.5]),
    ])
    train_dataset = SimpleImageDataset(
        txt_file_path=args.train_image_path,
        transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    print(f"Size of train_dataloader: {len(train_dataloader)}")

    # 3) VAEモデル読み込み
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path)

    # xFormers があるなら有効化
    if args.xformers:
        try:
            vae.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory efficient attention for VAE.")
        except Exception as e:
            print(f"Failed to enable xFormers: {e}")

    # gradient checkpointing を有効に
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # 4) Encoderを凍結し、Decoderだけ勾配を流す
    vae.encoder.requires_grad_(False)
    vae.encoder.eval()
    vae.decoder.train()

    # 5) Optimizer: decoder だけパラメータ更新
    if args.optimizer_type == "AdamW8bit":
        optimizer = bnb.optim.AdamW8bit(vae.decoder.parameters(), lr=args.lr)
        print("Using 8bit Adam optimizer.")
    else:
        optimizer = torch.optim.AdamW(vae.decoder.parameters(), lr=args.lr)
        print("Using standard AdamW optimizer.")

    # 6) LPIPSモデル（推論専用 -> eval）
    lpips_fn = lpips.LPIPS(net="alex")
    lpips_fn.eval()

    # 7) Accelerateにより prepare
    vae, optimizer, train_dataloader, lpips_fn = accelerator.prepare(
        vae, optimizer, train_dataloader, lpips_fn
    )

    # 8) 学習ループ
    for epoch in tqdm(range(args.epochs)):
        accumulation_steps = args.accumulation_steps  #勾配を貯めてからupdateする
        for step, images in enumerate(train_dataloader):
            with torch.no_grad():
                posterior = vae.encode(images).latent_dist
                z = posterior.sample()

            recon_images = vae.decode(z).sample

            mse_loss = patch_based_mse_loss(images, recon_images)
            lpips_loss = patch_based_lpips_loss(lpips_fn, images, recon_images)
            loss = 10 * mse_loss + 0.5 * lpips_loss

            # Backprop
            loss = loss / accumulation_steps  # 累積勾配のためにスケーリング
            accelerator.backward(loss)

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch}, step {step}: loss={loss.item():.4f} "
                      f"(mse={mse_loss.item():.4f}, lpips={lpips_loss.item():.4f})")    

            # Reconstruction Image の保存
            if step % 10 == 0:                       
                recon_img_cpu = recon_images[0].detach().float()
                recon_img_cpu = torch.clamp(recon_img_cpu, 0.0, 1.0)
                pil_img = transforms.ToPILImage()(recon_img_cpu.cpu())
                os.makedirs("recon_samples", exist_ok=True)
                save_path = f"recon_samples/recon_epoch{epoch}_step{step}.png"
                pil_img.save(save_path)
                print(f"Saved reconstruction to {save_path}")

    # 9) 学習済みVAEを保存
    unwrapped_vae = accelerator.unwrap_model(vae)
    unwrapped_vae.save_pretrained(args.model_save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="vae_models")
    parser.add_argument("--train_image_path", type=str, default="train_image_path.txt")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--optimizer_type", type=str, default="AdamW8bit")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_save_name", type=str, default="my_vae_decoder_only")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    main(args)
