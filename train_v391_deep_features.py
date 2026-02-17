#!/usr/bin/env python3
"""
V3.91 深度学习特征版本

核心思路：
1. 用自编码器提取非线性压缩特征
2. 用MLP学习因子之间的非线性交互
3. 原始因子 + 深度特征 组合训练

"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, rankdata
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置日志
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/v391_deep_features_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class FactorAutoencoder(nn.Module):
    """因子自编码器 - 提取非线性压缩特征"""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, latent_dim),
            nn.Tanh()  # 限制输出范围
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x):
        return self.encoder(x)


class FactorInteractionNet(nn.Module):
    """因子交互网络 - 学习因子间非线性关系"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # 输出层 - 预测收益排名
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def extract_features(self, x):
        """提取最后一层隐藏层的特征"""
        # 前向传播到倒数第二层
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x


class DeepFeatureV391Trainer:
    """深度学习特征V3.91训练器"""

    PERIOD_WEIGHTS = {'5d': 0.40, '10d': 0.35, '15d': 0.25}

    def __init__(self, db_path: str = 'data_adapter/stock_data.db'):
        self.db_path = db_path
        self.autoencoder = None
        self.interaction_net = None
        self.scaler = RobustScaler()

    def load_data(self, start_date: str = '2022-01-01') -> Dict:
        """加载数据"""
        import sqlite3

        logger.info("=" * 80)
        logger.info(f"📥 加载数据 (从 {start_date} 开始)")
        logger.info("=" * 80)

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT code, trade_date, features_json as features,
                   label_5d as future_return_5d,
                   label_10d as future_return_10d,
                   label_15d as future_return_15d
            FROM v39_feature_cache
            WHERE trade_date >= ?
            AND features_json IS NOT NULL
            AND label_5d IS NOT NULL
        """

        df = pd.read_sql_query(query, conn, params=[start_date])
        conn.close()

        logger.info(f"✅ 加载 {len(df):,} 个样本")

        # 解析特征
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                features = json.loads(row['features'])
                if isinstance(features, dict):
                    features_list.append(features)
                    valid_indices.append(idx)
            except:
                continue

        df_valid = df.loc[valid_indices].copy()
        feature_df = pd.DataFrame(features_list)

        # 获取数值特征
        numeric_cols = [c for c in feature_df.columns
                       if feature_df[c].dtype in ['float64', 'int64']]
        logger.info(f"📊 使用 {len(numeric_cols)} 个原始特征")

        for col in numeric_cols:
            df_valid[col] = feature_df[col].values

        # 按时间排序
        df_valid = df_valid.sort_values('trade_date').reset_index(drop=True)

        # 时序分割
        unique_dates = df_valid['trade_date'].unique()
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]

        train_mask = df_valid['trade_date'] < split_date
        val_mask = df_valid['trade_date'] >= split_date

        logger.info(f"📅 时间范围: {unique_dates[0]} ~ {unique_dates[-1]}")
        logger.info(f"📊 训练集: {train_mask.sum():,} 样本")
        logger.info(f"📊 验证集: {val_mask.sum():,} 样本")

        return {
            'df': df_valid,
            'feature_names': numeric_cols,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'split_date': split_date
        }

    def train_autoencoder(self, X_train: np.ndarray,
                          latent_dim: int = 16,
                          epochs: int = 50,
                          batch_size: int = 1024) -> FactorAutoencoder:
        """训练自编码器"""

        logger.info(f"\n🔷 训练自编码器 (latent_dim={latent_dim})")

        input_dim = X_train.shape[1]
        model = FactorAutoencoder(input_dim, latent_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 准备数据
        X_tensor = torch.FloatTensor(X_train).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                reconstructed, latent = model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return model

    def train_interaction_net(self, X_train: np.ndarray, y_train: np.ndarray,
                               hidden_dims: List[int] = [64, 32, 16],
                               epochs: int = 100,
                               batch_size: int = 1024) -> FactorInteractionNet:
        """训练因子交互网络"""

        logger.info(f"\n🔷 训练因子交互网络 (dims={hidden_dims})")

        input_dim = X_train.shape[1]
        model = FactorInteractionNet(input_dim, hidden_dims).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        # 标签转换为排名
        y_rank = (rankdata(y_train) - 1) / (len(y_train) - 1) - 0.5
        y_rank = y_rank.astype(np.float32)

        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_rank).unsqueeze(1).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x, y = batch
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return model

    def extract_deep_features(self, X: np.ndarray,
                               autoencoder: FactorAutoencoder,
                               interaction_net: FactorInteractionNet) -> np.ndarray:
        """提取深度特征"""

        X_tensor = torch.FloatTensor(X).to(device)

        autoencoder.eval()
        interaction_net.eval()

        with torch.no_grad():
            # 自编码器潜在特征
            latent_features = autoencoder.encode(X_tensor).cpu().numpy()
            # 交互网络隐藏层特征
            interaction_features = interaction_net.extract_features(X_tensor).cpu().numpy()

        # 组合: 原始特征 + 自编码特征 + 交互特征
        combined = np.hstack([X, latent_features, interaction_features])

        return combined

    def train_period(self, period: str, data: Dict,
                     latent_dim: int = 16,
                     ae_epochs: int = 50,
                     net_epochs: int = 100) -> Dict:
        """训练单个周期"""

        logger.info(f"\n{'='*60}")
        logger.info(f"🔹 训练 {period} 周期 (深度特征)")
        logger.info(f"{'='*60}")

        df = data['df']
        feature_names = data['feature_names']
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        label_col = f'future_return_{period}'

        # 提取数据
        X_all = df[feature_names].values
        y_all = df[label_col].values
        X_all = np.nan_to_num(X_all, nan=0.0).astype(np.float32)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_val = X_all[val_mask]
        y_val = y_all[val_mask]

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)

        # Step 1: 训练自编码器
        autoencoder = self.train_autoencoder(
            X_train_scaled, latent_dim, ae_epochs
        )

        # Step 2: 训练因子交互网络
        interaction_net = self.train_interaction_net(
            X_train_scaled, y_train, epochs=net_epochs
        )

        # Step 3: 提取深度特征
        logger.info("\n🔷 提取深度特征")
        X_train_deep = self.extract_deep_features(
            X_train_scaled, autoencoder, interaction_net
        )
        X_val_deep = self.extract_deep_features(
            X_val_scaled, autoencoder, interaction_net
        )

        logger.info(f"  原始特征维度: {X_train_scaled.shape[1]}")
        logger.info(f"  深度特征维度: {X_train_deep.shape[1]}")

        # Step 4: 用组合特征训练最终模型
        logger.info("\n🔷 训练最终Ridge模型")

        # 再次标准化组合特征
        deep_scaler = RobustScaler()
        X_train_final = deep_scaler.fit_transform(X_train_deep)
        X_val_final = deep_scaler.transform(X_val_deep)

        y_train_rank = (rankdata(y_train) - 1) / (len(y_train) - 1) - 0.5

        final_model = Ridge(alpha=10.0, random_state=42)
        final_model.fit(X_train_final, y_train_rank)

        # 预测
        train_pred = final_model.predict(X_train_final)
        val_pred = final_model.predict(X_val_final)

        # 评估
        ic_train = spearmanr(train_pred, y_train)[0]
        ic_val = spearmanr(val_pred, y_val)[0]
        ic_gap = (ic_train - ic_val) / ic_train * 100 if ic_train > 0 else 0

        logger.info(f"\n📊 {period} 周期结果:")
        logger.info(f"  训练IC: {ic_train:.4f}")
        logger.info(f"  验证IC: {ic_val:.4f}")
        logger.info(f"  IC差距: {ic_gap:.1f}%")

        # Top收益
        top_n = 100
        top_indices = np.argsort(val_pred)[-top_n:]
        top_return = y_val[top_indices].mean() * 100
        logger.info(f"  Top{top_n}收益: {top_return:.2f}%")

        # 方向准确率
        direction_correct = np.sum((val_pred > 0) == (y_val > 0)) / len(y_val)
        logger.info(f"  方向准确率: {direction_correct:.2%}")

        return {
            'autoencoder': autoencoder,
            'interaction_net': interaction_net,
            'scaler': self.scaler,
            'deep_scaler': deep_scaler,
            'final_model': final_model,
            'train_ic': ic_train,
            'val_ic': ic_val,
            'ic_gap': ic_gap,
            'top_return': top_return,
            'direction_accuracy': direction_correct
        }

    def train_all_periods(self, start_date: str = '2022-01-01') -> Dict:
        """训练所有周期"""

        logger.info("=" * 80)
        logger.info("🚀 V3.91 深度学习特征版本训练")
        logger.info("=" * 80)
        logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"数据起始日期: {start_date}")
        logger.info("=" * 80)

        # 加载数据
        data = self.load_data(start_date)

        # 训练各周期
        period_results = {}
        for period in ['5d', '10d', '15d']:
            result = self.train_period(period, data)
            period_results[period] = result

        # 综合评估
        composite_train_ic = sum(
            period_results[p]['train_ic'] * self.PERIOD_WEIGHTS[p]
            for p in ['5d', '10d', '15d']
        )
        composite_val_ic = sum(
            period_results[p]['val_ic'] * self.PERIOD_WEIGHTS[p]
            for p in ['5d', '10d', '15d']
        )
        composite_gap = (composite_train_ic - composite_val_ic) / composite_train_ic * 100

        logger.info("\n" + "=" * 80)
        logger.info("📊 综合评估")
        logger.info("=" * 80)
        logger.info(f"综合训练IC: {composite_train_ic:.4f}")
        logger.info(f"综合验证IC: {composite_val_ic:.4f}")
        logger.info(f"综合IC差距: {composite_gap:.1f}%")

        if composite_gap < 20:
            logger.info("✅ 过拟合已完全控制! (差距 < 20%)")
        elif composite_gap < 30:
            logger.info("✅ 过拟合控制良好! (差距 < 30%)")
        elif composite_gap < 50:
            logger.info("⚠️ 轻度过拟合 (差距 30-50%)")
        else:
            logger.info("❌ 过拟合较严重 (差距 > 50%)")

        # 计算评分
        score = self.calculate_score(composite_val_ic, composite_gap, period_results)
        logger.info(f"\n🎯 模型评分: {score:.1f}/100")

        return {
            'period_results': period_results,
            'composite_train_ic': composite_train_ic,
            'composite_val_ic': composite_val_ic,
            'composite_gap': composite_gap,
            'feature_names': data['feature_names'],
            'start_date': start_date,
            'score': score
        }

    def calculate_score(self, val_ic: float, gap: float, period_results: Dict) -> float:
        """计算综合评分"""

        # IC评分 (0-40分)
        if val_ic >= 0.15:
            ic_score = 40
        elif val_ic >= 0.10:
            ic_score = 30 + (val_ic - 0.10) / 0.05 * 10
        elif val_ic >= 0.05:
            ic_score = 20 + (val_ic - 0.05) / 0.05 * 10
        elif val_ic >= 0:
            ic_score = val_ic / 0.05 * 20
        else:
            ic_score = 0

        # 过拟合控制评分 (0-30分)
        if gap <= 20:
            gap_score = 30
        elif gap <= 30:
            gap_score = 25
        elif gap <= 50:
            gap_score = 15
        else:
            gap_score = max(0, 15 - (gap - 50) / 10)

        # Top收益评分 (0-30分)
        avg_top_return = np.mean([r['top_return'] for r in period_results.values()])
        if avg_top_return >= 3:
            return_score = 30
        elif avg_top_return >= 2:
            return_score = 25
        elif avg_top_return >= 1:
            return_score = 20
        elif avg_top_return >= 0:
            return_score = 10
        else:
            return_score = 0

        return ic_score + gap_score + return_score

    def save_model(self, results: Dict, version: str = 'deep_features'):
        """保存模型"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = 'models/v391'
        os.makedirs(model_dir, exist_ok=True)

        filename = f'v391_{version}_{timestamp}.pkl'
        filepath = os.path.join(model_dir, filename)

        # 注意：PyTorch模型需要特殊处理
        save_data = {
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'period_weights': self.PERIOD_WEIGHTS,
            'feature_names': results['feature_names'],
            'start_date': results['start_date'],
            'metrics': {
                'composite_train_ic': results['composite_train_ic'],
                'composite_val_ic': results['composite_val_ic'],
                'composite_gap': results['composite_gap'],
                'score': results['score']
            },
            'architecture': 'Autoencoder + InteractionNet + Ridge',
            'period_models': {}
        }

        for period, period_result in results['period_results'].items():
            # 保存PyTorch模型的state_dict
            save_data['period_models'][period] = {
                'autoencoder_state': period_result['autoencoder'].state_dict(),
                'interaction_net_state': period_result['interaction_net'].state_dict(),
                'scaler': period_result['scaler'],
                'deep_scaler': period_result['deep_scaler'],
                'final_model': period_result['final_model'],
                'train_ic': period_result['train_ic'],
                'val_ic': period_result['val_ic'],
                'top_return': period_result['top_return']
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"\n✅ 模型已保存: {filepath}")

        # 更新版本历史
        version_file = os.path.join(model_dir, 'VERSION_HISTORY.json')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_history = json.load(f)
        else:
            version_history = {'versions': []}

        version_history['updated'] = datetime.now().isoformat()
        version_history['versions'].insert(0, {
            'filename': filename,
            'version': f'v3.91-{version}',
            'timestamp': timestamp,
            'start_date': results['start_date'],
            'features': len(results['feature_names']),
            'composite_train_ic': round(results['composite_train_ic'], 4),
            'composite_val_ic': round(results['composite_val_ic'], 4),
            'composite_gap': f"{results['composite_gap']:.1f}%",
            'score': round(results['score'], 1),
            'architecture': 'Deep Learning Features',
            'description': "自编码器+交互网络提取非线性特征"
        })

        with open(version_file, 'w') as f:
            json.dump(version_history, f, indent=2)

        return filepath


def main():
    """主函数"""
    trainer = DeepFeatureV391Trainer()

    # 使用全量数据
    results = trainer.train_all_periods(start_date='2022-01-01')

    trainer.save_model(results, version='deep_features')

    logger.info("\n" + "=" * 80)
    logger.info("🎉 训练完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
