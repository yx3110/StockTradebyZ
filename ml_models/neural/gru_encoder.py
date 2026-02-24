#!/usr/bin/env python3
"""
GRU Temporal Encoder for Stock Price Sequences

从 60 天原始 OHLCV 序列中学习手工特征无法捕捉的时序模式,
输出 16 维嵌入向量作为 GBDT 模型的补充特征.

Architecture:
  Input:  (batch, 60, 6) — [open/close[-1]-1, high/close[-1]-1, low/close[-1]-1,
                             close/close[-1]-1, log_volume_zscore, pct_change]
  BatchNorm1d(6) → GRU(6→32, 2 layers, dropout=0.1) → Linear(32→16) → embedding
  Prediction head: Linear(16→1) — trains on label_5d, discarded at inference

Training:
  - Loss: MSE on per-date z-scored label_5d
  - Optimizer: AdamW(lr=1e-3, wd=1e-4), cosine annealing
  - Batch: 4096, Epochs: 20, early stop by val IC
  - Device: MPS (Apple Silicon) with CPU fallback
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from datetime import datetime
import json
import argparse

# Fix OMP conflict between LightGBM and PyTorch on macOS
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. GRU encoder unavailable.")

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 16
INPUT_DIM = 6  # OHLCV normalized + pct_change


def get_device():
    """Get best available device: MPS > CUDA > CPU"""
    if not HAS_TORCH:
        return None
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def prebuild_all_sequences(codes_dates: pd.DataFrame, ohlcv_data: Dict[str, pd.DataFrame]) -> np.ndarray:
    """Pre-build all (N, 60, 6) sequences in one pass per stock using vectorized operations.

    This avoids per-sample DataFrame filtering which is O(N*T) and extremely slow.
    Instead, we iterate per stock and use binary search to find date positions.

    Returns: np.ndarray of shape (N, SEQUENCE_LENGTH, INPUT_DIM)
    """
    N = len(codes_dates)
    all_seqs = np.zeros((N, SEQUENCE_LENGTH, INPUT_DIM), dtype=np.float32)

    # Build index: group sample indices by stock code
    code_groups = codes_dates.groupby('code').groups  # {code: [idx1, idx2, ...]}

    # For each stock, pre-convert date arrays to enable fast lookup
    done = 0
    total_stocks = len(code_groups)

    for code, indices in code_groups.items():
        if code not in ohlcv_data:
            done += 1
            continue

        df = ohlcv_data[code]
        dates_arr = df['trade_date'].values  # sorted ascending
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values

        # Get target dates for this stock's samples
        sample_dates = codes_dates.iloc[indices]['trade_date'].values

        for i, (sample_idx, target_date) in enumerate(zip(indices, sample_dates)):
            # Binary search for position of target_date
            pos = np.searchsorted(dates_arr, target_date, side='right')  # first index > target_date
            # pos is the number of dates <= target_date

            if pos < 2:
                continue  # need at least 2 points

            # Extract window: up to 61 points ending at pos
            start_pos = max(0, pos - SEQUENCE_LENGTH - 1)
            end_pos = pos

            w_opens = opens[start_pos:end_pos]
            w_highs = highs[start_pos:end_pos]
            w_lows = lows[start_pos:end_pos]
            w_closes = closes[start_pos:end_pos]
            w_volumes = volumes[start_pos:end_pos]

            if len(w_closes) < 2:
                continue

            # Normalize relative to previous close
            prev_close = w_closes[:-1]
            curr_open = w_opens[1:]
            curr_high = w_highs[1:]
            curr_low = w_lows[1:]
            curr_close = w_closes[1:]
            curr_vol = w_volumes[1:]

            n = len(prev_close)
            start = SEQUENCE_LENGTH - n
            safe_prev = np.where(prev_close > 0, prev_close, 1.0)

            all_seqs[sample_idx, start:, 0] = curr_open / safe_prev - 1
            all_seqs[sample_idx, start:, 1] = curr_high / safe_prev - 1
            all_seqs[sample_idx, start:, 2] = curr_low / safe_prev - 1
            all_seqs[sample_idx, start:, 3] = curr_close / safe_prev - 1
            all_seqs[sample_idx, start:, 5] = np.where(safe_prev > 0,
                                                        (curr_close - safe_prev) / safe_prev, 0)

            # Volume: log z-score
            log_vol = np.log1p(curr_vol.astype(np.float64))
            vol_mean = log_vol.mean()
            vol_std = log_vol.std()
            if vol_std > 1e-8:
                all_seqs[sample_idx, start:, 4] = (log_vol - vol_mean) / vol_std

        done += 1
        if done % 500 == 0:
            logger.info(f"  序列构建进度: {done}/{total_stocks} 股票")

    # Clip extreme values
    np.clip(all_seqs, -0.5, 0.5, out=all_seqs)
    logger.info(f"  序列构建完成: {N} 个样本")
    return all_seqs


class StockSequenceDataset(Dataset):
    """股票 OHLCV 60天序列数据集 (预构建版本)"""

    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Args:
            sequences: Pre-built (N, 60, 6) array
            labels: Optional target labels (length N)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx])
        if self.labels is not None:
            return seq, torch.FloatTensor([self.labels[idx]])
        return seq


if HAS_TORCH:
    class TemporalGRUEncoder(nn.Module):
        """GRU 时序编码器"""

        def __init__(self, input_dim=INPUT_DIM, hidden_dim=32, num_layers=2,
                     embedding_dim=EMBEDDING_DIM, dropout=0.1):
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim

            # Input normalization
            self.bn = nn.BatchNorm1d(input_dim)

            # GRU encoder
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

            # Embedding projection
            self.embed_proj = nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.Tanh(),
            )

            # Prediction head (trained, discarded at inference)
            self.pred_head = nn.Linear(embedding_dim, 1)

        def forward(self, x, return_embedding=False):
            """
            Args:
                x: (batch, seq_len, input_dim)
                return_embedding: if True, return embedding instead of prediction

            Returns:
                prediction (batch, 1) or embedding (batch, embedding_dim)
            """
            batch_size, seq_len, _ = x.shape

            # BatchNorm: need (batch, features, seq) format
            x_bn = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

            # GRU: take last hidden state
            _, h_n = self.gru(x_bn)  # h_n: (num_layers, batch, hidden)
            last_hidden = h_n[-1]     # (batch, hidden)

            # Embedding
            embedding = self.embed_proj(last_hidden)  # (batch, embedding_dim)

            if return_embedding:
                return embedding

            # Prediction
            pred = self.pred_head(embedding)  # (batch, 1)
            return pred

        def get_embedding(self, x):
            """Extract embedding only (for inference)"""
            self.eval()
            with torch.no_grad():
                return self.forward(x, return_embedding=True)


def load_ohlcv_data(db_path: str = DB_PATH,
                    start_date: str = None,
                    end_date: str = None) -> Dict[str, pd.DataFrame]:
    """批量加载所有 A 股 OHLCV 数据"""
    logger.info("加载 OHLCV 数据...")

    conn = sqlite3.connect(db_path)

    date_filter = ""
    if start_date:
        date_filter += f" AND q.trade_date >= '{start_date}'"
    if end_date:
        date_filter += f" AND q.trade_date <= '{end_date}'"

    query = f"""
    SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type = 'A股' AND q.volume > 0
    {date_filter}
    ORDER BY s.code, q.trade_date
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"  加载 {len(df):,} 条 OHLCV 记录")

    # Group by code
    ohlcv_data = {}
    for code, group in df.groupby('code'):
        ohlcv_data[code] = group.reset_index(drop=True)

    logger.info(f"  {len(ohlcv_data)} 只股票")
    return ohlcv_data


def load_training_labels(db_path: str = DB_PATH,
                         start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
    """从 v39_feature_cache 加载训练标签"""
    conn = sqlite3.connect(db_path)

    date_filter = ""
    if start_date:
        date_filter += f" AND trade_date >= '{start_date}'"
    if end_date:
        date_filter += f" AND trade_date <= '{end_date}'"

    query = f"""
    SELECT code, trade_date, label_5d
    FROM v39_feature_cache
    WHERE label_5d IS NOT NULL {date_filter}
    ORDER BY trade_date, code
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Per-date z-score labels
    df['label_zscore'] = df.groupby('trade_date')['label_5d'].transform(
        lambda x: (x - x.median()) / (x.std() + 1e-8)
    ).clip(-3, 3)

    return df


def train_gru_encoder(db_path: str = DB_PATH,
                      start_date: str = '2020-01-01',
                      end_date: str = None,
                      epochs: int = 20,
                      batch_size: int = 4096,
                      lr: float = 1e-3,
                      save_dir: str = None) -> str:
    """训练 GRU 编码器"""
    if not HAS_TORCH:
        raise ImportError("PyTorch required for GRU training. Install: pip install torch")

    device = get_device()
    logger.info(f"Training on device: {device}")

    # 1. Load data
    # Need OHLCV from 60 days before start_date
    ohlcv_start = pd.to_datetime(start_date) - pd.Timedelta(days=120)
    ohlcv_data = load_ohlcv_data(db_path, ohlcv_start.strftime('%Y-%m-%d'), end_date)
    labels_df = load_training_labels(db_path, start_date, end_date)

    logger.info(f"训练标签: {len(labels_df):,} 条")

    # 2. Temporal split
    unique_dates = np.sort(labels_df['trade_date'].unique())
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * 0.80)
    val_end_idx = int(n_dates * 0.90)

    train_dates = set(unique_dates[:train_end_idx])
    val_dates = set(unique_dates[train_end_idx:val_end_idx])
    test_dates = set(unique_dates[val_end_idx:])

    train_mask = labels_df['trade_date'].isin(train_dates)
    val_mask = labels_df['trade_date'].isin(val_dates)
    test_mask = labels_df['trade_date'].isin(test_dates)

    train_df = labels_df[train_mask].reset_index(drop=True)
    val_df = labels_df[val_mask].reset_index(drop=True)
    test_df = labels_df[test_mask].reset_index(drop=True)

    logger.info(f"  训练: {len(train_df):,}, 验证: {len(val_df):,}, 测试: {len(test_df):,}")

    # 3. Pre-build all sequences (vectorized, ~5-10 min for 7M samples)
    logger.info("预构建训练序列...")
    import time as _time
    t0 = _time.time()
    train_seqs = prebuild_all_sequences(train_df[['code', 'trade_date']], ohlcv_data)
    logger.info(f"  训练序列: {train_seqs.shape}, 耗时 {_time.time()-t0:.1f}秒")

    t0 = _time.time()
    val_seqs = prebuild_all_sequences(val_df[['code', 'trade_date']], ohlcv_data)
    logger.info(f"  验证序列: {val_seqs.shape}, 耗时 {_time.time()-t0:.1f}秒")

    # Create datasets from pre-built arrays
    train_dataset = StockSequenceDataset(train_seqs, train_df['label_zscore'].values)
    val_dataset = StockSequenceDataset(val_seqs, val_df['label_zscore'].values)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

    # 4. Model
    model = TemporalGRUEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Training loop
    best_val_ic = -1
    best_epoch = 0
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                val_preds.append(pred.cpu().numpy().flatten())
                val_labels.append(batch_y.numpy().flatten())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        # Calculate daily IC on validation set
        from scipy.stats import spearmanr
        val_ic, _ = spearmanr(val_preds, val_labels)

        # Per-date IC
        daily_ics = []
        for date in val_df['trade_date'].unique():
            mask = val_df['trade_date'].values == date
            if mask.sum() < 20:
                continue
            ic, _ = spearmanr(val_preds[mask], val_labels[mask])
            if not np.isnan(ic):
                daily_ics.append(ic)

        mean_daily_ic = np.mean(daily_ics) if daily_ics else 0
        ic_ir = mean_daily_ic / np.std(daily_ics) if daily_ics and np.std(daily_ics) > 0 else 0

        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_train_loss:.6f}, "
                   f"val_IC={val_ic:.4f}, daily_IC={mean_daily_ic:.4f}, ICIR={ic_ir:.4f}")

        # Early stopping by daily IC
        if mean_daily_ic > best_val_ic:
            best_val_ic = mean_daily_ic
            best_epoch = epoch + 1
            no_improve = 0
            # Save best model
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch})")
                break

    # 6. Load best model and test
    model.load_state_dict(best_state)
    model.eval()

    # Test IC
    logger.info("预构建测试序列...")
    t0 = _time.time()
    test_seqs = prebuild_all_sequences(test_df[['code', 'trade_date']], ohlcv_data)
    logger.info(f"  测试序列: {test_seqs.shape}, 耗时 {_time.time()-t0:.1f}秒")
    test_dataset = StockSequenceDataset(test_seqs, test_df['label_zscore'].values)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_preds = []
    test_labels_arr = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            test_preds.append(pred.cpu().numpy().flatten())
            test_labels_arr.append(batch_y.numpy().flatten())

    test_preds = np.concatenate(test_preds)
    test_labels_arr = np.concatenate(test_labels_arr)

    test_ic, _ = spearmanr(test_preds, test_labels_arr)
    daily_test_ics = []
    for date in test_df['trade_date'].unique():
        mask = test_df['trade_date'].values == date
        if mask.sum() < 20:
            continue
        ic, _ = spearmanr(test_preds[mask], test_labels_arr[mask])
        if not np.isnan(ic):
            daily_test_ics.append(ic)

    test_daily_ic = np.mean(daily_test_ics) if daily_test_ics else 0
    test_icir = test_daily_ic / np.std(daily_test_ics) if daily_test_ics and np.std(daily_test_ics) > 0 else 0

    logger.info(f"\n  测试集: IC={test_ic:.4f}, daily_IC={test_daily_ic:.4f}, ICIR={test_icir:.4f}")

    # 7. Save model
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'neural')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = save_path / f'gru_encoder_{timestamp}.pt'

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': INPUT_DIM,
            'hidden_dim': 32,
            'num_layers': 2,
            'embedding_dim': EMBEDDING_DIM,
            'dropout': 0.1,
            'sequence_length': SEQUENCE_LENGTH,
        },
        'training': {
            'best_epoch': best_epoch,
            'best_val_ic': best_val_ic,
            'test_ic': test_ic,
            'test_daily_ic': test_daily_ic,
            'test_icir': test_icir,
            'start_date': start_date,
            'end_date': end_date,
        }
    }, model_file)

    # Save latest link
    latest_file = save_path / 'gru_encoder_latest.pt'
    torch.save(torch.load(model_file, weights_only=False), latest_file)

    logger.info(f"模型已保存: {model_file}")
    return str(model_file)


def load_gru_model(model_path: str = None, device=None):
    """加载训练好的 GRU 模型"""
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    if model_path is None:
        model_path = str(PROJECT_ROOT / 'ml_models' / 'trained_models' / 'neural' /
                        'gru_encoder_latest.pt')

    if device is None:
        device = get_device()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = TemporalGRUEncoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        embedding_dim=config['embedding_dim'],
        dropout=config.get('dropout', 0.1),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Train GRU Temporal Encoder')
    parser.add_argument('--start-date', type=str, default='2020-01-01')
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_gru_encoder(
        start_date=args.start_date,
        end_date=args.end_date,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == '__main__':
    main()
