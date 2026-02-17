#!/usr/bin/env python3
"""
因子管理框架 - 统一管理因子定义、计算、存储和查询
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import sqlite3
import logging
from pathlib import Path
import json

class FactorManager:
    """因子管理器 - 中心化的因子管理"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """初始化因子管理器"""
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # 因子注册表
        self.factor_registry = {}
        
        # 初始化默认因子
        self._register_default_factors()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("FactorManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _register_default_factors(self):
        """注册默认因子"""
        
        # 技术因子
        self.register_factor(
            name="momentum_5d",
            category="technical",
            description="5日价格动量",
            dependencies=["close"],
            calculator=lambda df: df['close'].pct_change(5) * 100
        )
        
        self.register_factor(
            name="volatility_20d",
            category="technical", 
            description="20日波动率",
            dependencies=["price_change_pct"],
            calculator=lambda df: df['price_change_pct'].rolling(20).std()
        )
        
        self.register_factor(
            name="volume_ratio_20d",
            category="technical",
            description="20日成交量比率",
            dependencies=["volume"],
            calculator=lambda df: df['volume'] / df['volume'].rolling(20).mean()
        )
        
        # 市场因子
        self.register_factor(
            name="relative_strength",
            category="market",
            description="相对强度指数",
            dependencies=["stock_return", "market_return"],
            calculator=self._calculate_relative_strength
        )
        
        # 基本面因子
        self.register_factor(
            name="pe_percentile",
            category="fundamental",
            description="PE历史百分位",
            dependencies=["pe_ttm"],
            calculator=lambda df: df['pe_ttm'].rank(pct=True)
        )
    
    def register_factor(self, name: str, category: str, 
                       description: str, dependencies: List[str],
                       calculator: Callable, version: str = "1.0"):
        """注册新因子"""
        
        self.factor_registry[name] = {
            "category": category,
            "description": description,
            "dependencies": dependencies,
            "calculator": calculator,
            "version": version,
            "created_at": datetime.now()
        }
        
        # 同时保存到数据库
        self._save_factor_definition(name, category, description, 
                                    dependencies, version)
        
        self.logger.info(f"注册因子: {name} ({category})")
    
    def _save_factor_definition(self, name: str, category: str,
                               description: str, dependencies: List[str],
                               version: str):
        """保存因子定义到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='factor_definitions'
            """)
            
            if cursor.fetchone():
                cursor.execute("""
                    INSERT OR REPLACE INTO factor_definitions 
                    (factor_name, factor_category, factor_type, description, 
                     dependencies, version, created_date)
                    VALUES (?, ?, 'numeric', ?, ?, ?, ?)
                """, (
                    name, category, description,
                    json.dumps(dependencies), version,
                    datetime.now().date()
                ))
                conn.commit()
    
    def _calculate_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """计算相对强度"""
        return (df['stock_return'].rolling(20).mean() - 
                df['market_return'].rolling(20).mean()) * 100
    
    def get_factor_data(self, security_codes: List[str], 
                       factor_names: List[str],
                       start_date: str, 
                       end_date: str) -> pd.DataFrame:
        """获取因子数据"""
        
        # 构建查询
        factor_tables = {
            'technical': 'technical_factors',
            'market': 'market_factors',
            'fundamental': 'fundamental_factors'
        }
        
        queries = []
        for factor_name in factor_names:
            if factor_name in self.factor_registry:
                category = self.factor_registry[factor_name]['category']
                table = factor_tables.get(category)
                
                if table:
                    queries.append(f"""
                        SELECT s.code, t.trade_date, t.{factor_name}
                        FROM {table} t
                        JOIN securities s ON t.security_id = s.id
                        WHERE s.code IN ({','.join(['?']*len(security_codes))})
                        AND t.trade_date BETWEEN ? AND ?
                    """)
        
        # 执行查询并合并结果
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for query in queries:
                params = security_codes + [start_date, end_date]
                df = pd.read_sql_query(query, conn, params=params)
                results.append(df)
        
        if results:
            # 合并所有因子数据
            merged = results[0]
            for df in results[1:]:
                merged = pd.merge(merged, df, on=['code', 'trade_date'], how='outer')
            
            return merged
        
        return pd.DataFrame()
    
    def calculate_composite_score(self, security_code: str, 
                                 date: str,
                                 weights: Optional[Dict[str, float]] = None) -> float:
        """计算综合评分"""
        
        # 默认权重
        if weights is None:
            weights = {
                'momentum_5d': 0.2,
                'volatility_20d': -0.1,
                'volume_ratio_20d': 0.15,
                'relative_strength': 0.25,
                'pe_percentile': -0.1
            }
        
        # 获取因子数据
        factor_data = self.get_factor_data(
            [security_code], 
            list(weights.keys()),
            date, date
        )
        
        if factor_data.empty:
            return 0.0
        
        # 计算加权得分
        score = 0.0
        row = factor_data.iloc[0]
        
        for factor, weight in weights.items():
            if factor in row and not pd.isna(row[factor]):
                # 标准化到0-1
                normalized = self._normalize_factor(row[factor], factor)
                score += normalized * weight
        
        return max(0, min(100, score * 100))
    
    def _normalize_factor(self, value: float, factor_name: str) -> float:
        """标准化因子值到0-1范围"""
        
        # 这里可以根据历史数据计算分位数
        # 简化处理：使用预定义的范围
        ranges = {
            'momentum_5d': (-20, 20),
            'volatility_20d': (0, 5),
            'volume_ratio_20d': (0, 3),
            'relative_strength': (-10, 10),
            'pe_percentile': (0, 1)
        }
        
        if factor_name in ranges:
            min_val, max_val = ranges[factor_name]
            normalized = (value - min_val) / (max_val - min_val)
            return max(0, min(1, normalized))
        
        return 0.5
    
    def get_factor_correlation_matrix(self, factor_names: List[str],
                                     start_date: str,
                                     end_date: str) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        
        # 获取所有股票的因子数据
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT code FROM securities WHERE type = 'A股' LIMIT 100"
            stocks = pd.read_sql_query(query, conn)['code'].tolist()
        
        # 获取因子数据
        factor_data = self.get_factor_data(stocks, factor_names, 
                                          start_date, end_date)
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # 计算相关性矩阵
        correlation_matrix = factor_data[factor_names].corr()
        
        return correlation_matrix
    
    def analyze_factor_importance(self, target_returns: pd.Series,
                                 factor_data: pd.DataFrame) -> Dict[str, float]:
        """分析因子重要性（使用相关性）"""
        
        importance = {}
        
        for factor in factor_data.columns:
            if factor not in ['code', 'trade_date']:
                # 计算与目标收益的相关性
                correlation = factor_data[factor].corr(target_returns)
                importance[factor] = abs(correlation)
        
        # 排序
        importance = dict(sorted(importance.items(), 
                               key=lambda x: x[1], reverse=True))
        
        return importance
    
    def export_factor_data(self, output_path: str, 
                          start_date: str, 
                          end_date: str,
                          factor_names: Optional[List[str]] = None):
        """导出因子数据供神经网络训练"""
        
        self.logger.info(f"导出因子数据: {start_date} 到 {end_date}")
        
        # 获取所有股票
        with sqlite3.connect(self.db_path) as conn:
            stocks_query = "SELECT code FROM securities WHERE type = 'A股'"
            stocks = pd.read_sql_query(stocks_query, conn)['code'].tolist()
        
        # 如果没有指定因子，使用所有已注册的因子
        if factor_names is None:
            factor_names = list(self.factor_registry.keys())
        
        # 分批获取数据
        all_data = []
        batch_size = 100
        
        for i in range(0, len(stocks), batch_size):
            batch_stocks = stocks[i:i+batch_size]
            batch_data = self.get_factor_data(
                batch_stocks, factor_names,
                start_date, end_date
            )
            all_data.append(batch_data)
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 添加目标变量（未来收益）
        combined_data = self._add_target_returns(combined_data)
        
        # 保存到文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.endswith('.csv'):
            combined_data.to_csv(output_path, index=False)
        elif output_path.endswith('.parquet'):
            combined_data.to_parquet(output_path, index=False)
        else:
            combined_data.to_pickle(output_path)
        
        self.logger.info(f"导出完成: {len(combined_data)} 条记录保存到 {output_path}")
        
        return combined_data
    
    def _add_target_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加目标收益率（用于训练）"""
        
        # 获取未来收益率
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT 
                s.code,
                dq.trade_date,
                dq.close,
                LEAD(dq.close, 1) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_1d,
                LEAD(dq.close, 5) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_5d,
                LEAD(dq.close, 20) OVER (PARTITION BY s.code ORDER BY dq.trade_date) as close_20d
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股'
            """
            
            price_data = pd.read_sql_query(query, conn)
        
        # 计算收益率
        price_data['return_1d'] = (price_data['close_1d'] / price_data['close'] - 1) * 100
        price_data['return_5d'] = (price_data['close_5d'] / price_data['close'] - 1) * 100
        price_data['return_20d'] = (price_data['close_20d'] / price_data['close'] - 1) * 100
        
        # 合并到原数据
        df = pd.merge(
            df, 
            price_data[['code', 'trade_date', 'return_1d', 'return_5d', 'return_20d']],
            on=['code', 'trade_date'],
            how='left'
        )
        
        return df


class FactorPipeline:
    """因子处理管道 - 用于实时计算和批处理"""
    
    def __init__(self, factor_manager: FactorManager):
        """初始化处理管道"""
        self.factor_manager = factor_manager
        self.pipeline_steps = []
    
    def add_step(self, name: str, processor: Callable):
        """添加处理步骤"""
        self.pipeline_steps.append({
            'name': name,
            'processor': processor
        })
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行管道处理"""
        result = data.copy()
        
        for step in self.pipeline_steps:
            result = step['processor'](result)
        
        return result
    
    def create_training_dataset(self, start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """创建训练数据集"""
        
        # 导出因子数据
        factor_data = self.factor_manager.export_factor_data(
            'temp_factor_data.pkl',
            start_date,
            end_date
        )
        
        # 数据清洗
        factor_data = self._clean_data(factor_data)
        
        # 特征工程
        factor_data = self._engineer_features(factor_data)
        
        # 数据标准化
        factor_data = self._standardize_data(factor_data)
        
        return factor_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 删除缺失值过多的行
        df = df.dropna(thresh=len(df.columns) * 0.8)
        
        # 填充剩余缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        # 添加交互特征
        df['momentum_volatility'] = df['momentum_5d'] * df['volatility_20d']
        df['volume_momentum'] = df['volume_ratio_20d'] * df['momentum_5d']
        
        # 添加排名特征
        df['momentum_rank'] = df.groupby('trade_date')['momentum_5d'].rank(pct=True)
        df['volume_rank'] = df.groupby('trade_date')['volume_ratio_20d'].rank(pct=True)
        
        return df
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        from sklearn.preprocessing import StandardScaler
        
        # 选择数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        exclude_columns = ['return_1d', 'return_5d', 'return_20d']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # 标准化
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        return df


def main():
    """主函数：演示因子管理器使用"""
    
    # 初始化因子管理器
    manager = FactorManager()
    
    # 注册自定义因子
    manager.register_factor(
        name="custom_momentum",
        category="technical",
        description="自定义动量因子",
        dependencies=["close"],
        calculator=lambda df: df['close'].pct_change(10) * df['volume'].pct_change(10)
    )
    
    # 获取因子数据
    factor_data = manager.get_factor_data(
        security_codes=['000001.SZ', '000002.SZ'],
        factor_names=['momentum_5d', 'volatility_20d'],
        start_date='2025-01-01',
        end_date='2025-08-01'
    )
    
    print("因子数据示例:")
    print(factor_data.head())
    
    # 计算综合评分
    score = manager.calculate_composite_score(
        security_code='000001.SZ',
        date='2025-08-01'
    )
    
    print(f"\n综合评分: {score:.2f}")
    
    # 导出训练数据
    print("\n创建训练数据集...")
    pipeline = FactorPipeline(manager)
    training_data = pipeline.create_training_dataset(
        start_date='2024-01-01',
        end_date='2025-08-01'
    )
    
    print(f"训练数据集形状: {training_data.shape}")
    print(f"特征列: {list(training_data.columns)}")


if __name__ == "__main__":
    main()