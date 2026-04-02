#!/usr/bin/env python3
"""Quick local training for NC-SSM on GSC V2 (CPU-friendly)."""

import sys
import os
import time
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import nano_ssm

# ============================================================
# Config
# ============================================================
DATA_DIR = Path("C:/Users/jinho/Downloads/SmartEar-KWS-/data/SpeechCommands/speech_commands_v0.02")
SAVE_DIR = Path("C:/Users/jinho/Downloads/NanoMamba-Interspeech2026/checkpoints_full/NC-SSM")
MODEL_NAME = "ncssm"
EPOCHS = 30
BATCH_SIZE = 64
LR = 3e-3
SR = 16000
DURATION = 16000  # 1 second

LABELS_12 = ['yes', 'no', 'up', 'down', 'left', 'right',
             'on', 'off', 'stop', 'go', 'silence', 'unknown']
CORE_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right',
              'on', 'off', 'stop', 'go']

# ============================================================
# Dataset
# ============================================================
class GSCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='training', sr=16000, duration=16000):
        self.sr = sr
        self.duration = duration
        self.samples = []

        # Read split file
        if split == 'training':
            val_list = set()
            test_list = set()
            for f in ['validation_list.txt', 'testing_list.txt']:
                p = data_dir / f
                if p.exists():
                    with open(p) as fh:
                        lst = val_list if 'validation' in f else test_list
                        for line in fh:
                            lst.add(line.strip())

            for word_dir in sorted(data_dir.iterdir()):
                if not word_dir.is_dir() or word_dir.name.startswith('_'):
                    continue
                label = self._map_label(word_dir.name)
                label_idx = LABELS_12.index(label)
                for wav in word_dir.glob('*.wav'):
                    rel = f"{word_dir.name}/{wav.name}"
                    if rel not in val_list and rel not in test_list:
                        self.samples.append((str(wav), label_idx))
        else:
            list_file = data_dir / f'{split}_list.txt'
            if list_file.exists():
                with open(list_file) as f:
                    for line in f:
                        rel = line.strip()
                        if not rel:
                            continue
                        parts = rel.split('/')
                        word = parts[0]
                        label = self._map_label(word)
                        label_idx = LABELS_12.index(label)
                        wav_path = data_dir / rel
                        if wav_path.exists():
                            self.samples.append((str(wav_path), label_idx))

        print(f"  {split}: {len(self.samples)} samples")

    def _map_label(self, word):
        if word in CORE_WORDS:
            return word
        elif word == '_silence_' or word == 'silence':
            return 'silence'
        else:
            return 'unknown'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        import soundfile as sf
        wav_np, sr = sf.read(path, dtype='float32')
        wav = torch.from_numpy(wav_np)
        if wav.dim() > 1:
            wav = wav[:, 0]  # mono

        # Resample if needed
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)

        # Pad or trim
        if len(wav) < self.duration:
            wav = torch.nn.functional.pad(wav, (0, self.duration - len(wav)))
        else:
            wav = wav[:self.duration]

        return wav, label


# ============================================================
# Training
# ============================================================
def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = nano_ssm.create(MODEL_NAME)
    print(model.summary())
    model = model.to(device)

    # Datasets
    print("\nLoading GSC V2 dataset...")
    train_ds = GSCDataset(DATA_DIR, 'training')
    val_ds = GSCDataset(DATA_DIR, 'validation')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device == 'cuda'))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    print(f"\nTraining {MODEL_NAME} for {EPOCHS} epochs...")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        t0 = time.time()

        for i, (audio, labels) in enumerate(train_loader):
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                print(f"\r  Epoch {epoch}/{EPOCHS} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}", end='', flush=True)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                logits = model(audio)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        elapsed = time.time() - t0
        is_best = acc > best_acc

        if is_best:
            best_acc = acc
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'epoch': epoch,
                'val_acc': acc,
                'model_name': MODEL_NAME,
            }, SAVE_DIR / 'best.pt')

        marker = ' *BEST*' if is_best else ''
        print(f"\r  E{epoch:2d}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"acc={acc:.1f}% | lr={scheduler.get_last_lr()[0]:.1e} | "
              f"{elapsed:.0f}s{marker}")

    # Save final
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'epoch': EPOCHS,
        'val_acc': acc,
        'model_name': MODEL_NAME,
    }, SAVE_DIR / 'final.pt')

    print(f"\nDone! Best accuracy: {best_acc:.1f}%")
    print(f"Checkpoint saved: {SAVE_DIR / 'best.pt'}")


if __name__ == '__main__':
    train()
