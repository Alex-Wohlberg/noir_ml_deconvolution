import os, glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA

SKY, SCALE = 0.0, 0.7          # from your field audit

# --- pick the best available backend; falls back to CPU if no GPU ---
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
use_amp = (device == 'cuda')   # AMP/GradScaler are CUDA-only
print('using device:', device)

class FitsDataset(Dataset):
    def __init__(self, folder, sky=SKY, scale=SCALE):
        self.paths = sorted(glob.glob(os.path.join(folder, '*.fits')))
        assert self.paths, f'no .fits in {folder}'
        self.sky, self.scale = sky, scale
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        with fits.open(self.paths[idx]) as hdul:
            data = hdul[0].data.astype(np.float32)
        data = (data - self.sky) / self.scale
        data = np.clip(data * 2 - 1, -1, 1)        # -> [-1,1], outliers tamed
        if data.ndim == 2:
            data = data[None, ...]                  # (1, H, W)
        return torch.from_numpy(data)               # image only, no label

# --- model ---
print('[1/4] building model...')
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True, channels=1)
diffusion = GaussianDiffusion(model, image_size=64, timesteps=1000,
                              sampling_timesteps=250).to(device)
print('      model built and moved to', device)

# --- data ---
print('[2/4] loading dataset...')
ds = FitsDataset('/home/alex/noir_ml/mycode/patches/sharp/fits')
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
                pin_memory=(device == 'cuda'), drop_last=True)
print(f'      dataset ready: {len(ds)} patches, {len(dl)} batches/epoch')

# --- optimizer + EMA (the EMA weights are what PnP loads) ---
print('[3/4] setting up optimizer + EMA...')
opt = torch.optim.Adam(diffusion.parameters(), lr=8e-5)
ema = EMA(diffusion, beta=0.995, update_every=10).to(device)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
print('      optimizer, EMA, scaler ready')

# --- training loop ---
print('[4/4] starting training loop...')
train_num_steps, grad_accum = 700000, 2
step, data_iter = 0, iter(dl)
diffusion.train()
while step < train_num_steps:
    opt.zero_grad()
    for _ in range(grad_accum):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); batch = next(data_iter)
        batch = batch.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            loss = diffusion(batch) / grad_accum
        scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    ema.update()
    step += 1

    if step == 1:
        print(f'      first step complete (loss {loss.item()*grad_accum:.5f}) '
              f'— pipeline is working')
    if step % 100 == 0:
        print(f'step {step}/{train_num_steps}  loss {loss.item()*grad_accum:.5f}')
    if step % 10000 == 0:
        torch.save(ema.ema_model.state_dict(), f'diffusion_prior_step{step}.pt')
        print(f'      checkpoint saved at step {step}')

print('training loop finished, saving final weights...')
torch.save(ema.ema_model.state_dict(), 'diffusion_prior.pt')
print('done. saved diffusion_prior.pt') 