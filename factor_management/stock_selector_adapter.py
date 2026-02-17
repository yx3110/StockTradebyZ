#!/usr/bin/env python3
"""
选股系统与因子管理框架适配器
Stock Selector - Factor Management Framework Adapter

整合现有的4个选股策略与因子管理框架，提供统一的接口
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import importlib.util

# 导入因子管理框架
from factor_manager import FactorManager, FactorPipeline
from v2_factor_extractor import V2FactorExtractor
from v4_factor_extractor import V4FactorExtractor

class StockSelectorAdapter:
    """选股系统与因子管理框架适配器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.factor_manager = FactorManager(db_path)
        self.v2_extractor = V2FactorExtractor(db_path)
        self.v4_extractor = V4FactorExtractor(db_path)  # 🆕 添加V4提取器
        self.logger = self._setup_logger()
        
        # 导入选股器类
        self.selector_classes = self._import_selector_classes()
        
        # 注册选股策略相关因子
        self._register_selector_factors()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("StockSelectorAdapter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _import_selector_classes(self) -> Dict:
        """导入选股器类"""
        try:
            selector_path = Path("../stock_selctor/Selector.py")
            if not selector_path.exists():
                selector_path = Path("stock_selctor/Selector.py")
            
            if not selector_path.exists():
                self.logger.error(f"找不到选股器模块: {selector_path}")
                return {}
                
            spec = importlib.util.spec_from_file_location("Selector", selector_path)
            selector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(selector_module)
            
            selector_classes = {
                'BBIKDJSelector': selector_module.BBIKDJSelector,
                'BBIShortLongSelector': selector_module.BBIShortLongSelector, 
                'BreakoutVolumeKDJSelector': selector_module.BreakoutVolumeKDJSelector,
                'PeakKDJSelector': selector_module.PeakKDJSelector
            }
            
            self.logger.info(f"成功导入 {len(selector_classes)} 个选股器类")
            return selector_classes
            
        except Exception as e:
            self.logger.error(f"导入选股器失败: {e}")
            return {}
    
    def _register_selector_factors(self):
        """注册选股策略相关因子"""
        
        # 注册4个选股策略的因子
        self.factor_manager.register_factor(
            name="bbi_kdj_signal",
            category="strategy",
            description="少负战法信号(BBI+KDJ组合)",
            dependencies=["bbi", "K", "D", "J"],
            calculator=self._calculate_bbi_kdj_signal,
            version="1.0"
        )
        
        self.factor_manager.register_factor(
            name="bbi_short_long_signal", 
            category="strategy",
            description="补票战法信号(BBI短长期RSV)",
            dependencies=["bbi", "rsv_short", "rsv_long"],
            calculator=self._calculate_bbi_short_long_signal,
            version="1.0"
        )
        
        self.factor_manager.register_factor(
            name="breakout_volume_kdj_signal",
            category="strategy", 
            description="TePu战法信号(量价突破+KDJ)",
            dependencies=["volume", "close", "high", "K", "D", "J"],
            calculator=self._calculate_breakout_volume_kdj_signal,
            version="1.0"
        )
        
        self.factor_manager.register_factor(
            name="peak_kdj_signal",
            category="strategy",
            description="填坑战法信号(峰值检测+KDJ)",
            dependencies=["close", "K", "D", "J"],
            calculator=self._calculate_peak_kdj_signal,
            version="1.0"
        )
        
        # 注册综合策略信号
        self.factor_manager.register_factor(
            name="combined_strategy_signal",
            category="composite", 
            description="4策略综合信号",
            dependencies=["bbi_kdj_signal", "bbi_short_long_signal", 
                         "breakout_volume_kdj_signal", "peak_kdj_signal"],
            calculator=self._calculate_combined_strategy_signal,
            version="1.0"
        )
        
        self.logger.info("已注册5个选股策略因子")
    
    def _calculate_bbi_kdj_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算少负战法信号"""
        signals = []
        
        for i in range(len(df)):
            signal_strength = 0
            
            # BBI判断
            if i > 0 and 'bbi' in df.columns:
                if df['close'].iloc[i] > df['bbi'].iloc[i]:
                    signal_strength += 30  # 价格在BBI上方
                    
                if df['bbi'].iloc[i] > df['bbi'].iloc[i-1]:
                    signal_strength += 20  # BBI向上
            
            # KDJ判断 
            if 'K' in df.columns and 'D' in df.columns and 'J' in df.columns:
                K = df['K'].iloc[i] if pd.notna(df['K'].iloc[i]) else 50
                D = df['D'].iloc[i] if pd.notna(df['D'].iloc[i]) else 50
                J = df['J'].iloc[i] if pd.notna(df['J'].iloc[i]) else 50
                
                if K > D and J > K:  # 金叉且J值最高
                    signal_strength += 35
                elif K > D:  # 仅金叉
                    signal_strength += 20
                    
                if 20 < K < 80:  # K值在正常范围
                    signal_strength += 15
            
            signals.append(min(100, signal_strength))
        
        return pd.Series(signals, index=df.index)
    
    def _calculate_bbi_short_long_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算补票战法信号"""
        signals = []
        
        for i in range(len(df)):
            signal_strength = 0
            
            # BBI基础判断
            if i > 0 and 'bbi' in df.columns:
                if df['close'].iloc[i] > df['bbi'].iloc[i]:
                    signal_strength += 25
                    
            # RSV短期和长期对比（这里简化处理）
            if i >= 5:
                # 短期RSV (5日)
                low_5 = df['low'].iloc[i-4:i+1].min()
                high_5 = df['close'].iloc[i-4:i+1].max()
                rsv_short = (df['close'].iloc[i] - low_5) / (high_5 - low_5 + 1e-9) * 100
                
                # 长期RSV (15日)
                if i >= 15:
                    low_15 = df['low'].iloc[i-14:i+1].min()
                    high_15 = df['close'].iloc[i-14:i+1].max()  
                    rsv_long = (df['close'].iloc[i] - low_15) / (high_15 - low_15 + 1e-9) * 100
                    
                    if rsv_short > rsv_long and rsv_short > 50:
                        signal_strength += 40
                    elif rsv_short > 30:
                        signal_strength += 25
                        
                elif rsv_short > 50:
                    signal_strength += 35
                    
            signals.append(min(100, signal_strength))
        
        return pd.Series(signals, index=df.index)
    
    def _calculate_breakout_volume_kdj_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算TePu战法信号"""
        signals = []
        
        for i in range(len(df)):
            signal_strength = 0
            
            # 价格突破判断
            if i >= 10:
                current_close = df['close'].iloc[i]
                recent_high = df['high'].iloc[i-9:i].max()  # 过去10天最高价
                
                if current_close >= recent_high * 0.98:  # 接近或突破前期高点
                    signal_strength += 30
            
            # 成交量确认
            if i >= 5:
                current_volume = df['volume'].iloc[i]
                avg_volume = df['volume'].iloc[i-4:i].mean()  # 过去5天平均量
                
                if current_volume > avg_volume * 1.5:  # 放量1.5倍以上
                    signal_strength += 35
                elif current_volume > avg_volume * 1.2:  # 放量1.2倍以上
                    signal_strength += 20
            
            # KDJ确认
            if 'K' in df.columns and 'D' in df.columns:
                K = df['K'].iloc[i] if pd.notna(df['K'].iloc[i]) else 50
                D = df['D'].iloc[i] if pd.notna(df['D'].iloc[i]) else 50
                
                if K > D and K > 50:  # KDJ多头排列
                    signal_strength += 35
                elif K > 50:
                    signal_strength += 20
                    
            signals.append(min(100, signal_strength))
        
        return pd.Series(signals, index=df.index)
    
    def _calculate_peak_kdj_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算填坑战法信号"""
        signals = []
        
        for i in range(len(df)):
            signal_strength = 0
            
            # 峰值检测（简化版）
            if i >= 10:
                window = df['close'].iloc[i-9:i+1]
                current_price = window.iloc[-1]
                
                # 寻找前期高点
                peaks = []
                for j in range(1, len(window)-1):
                    if window.iloc[j] > window.iloc[j-1] and window.iloc[j] > window.iloc[j+1]:
                        peaks.append(window.iloc[j])
                
                if peaks:
                    highest_peak = max(peaks)
                    # 如果当前价格接近前期高点（填坑）
                    if current_price >= highest_peak * 0.95:
                        signal_strength += 40
                    elif current_price >= highest_peak * 0.90:
                        signal_strength += 25
            
            # KDJ确认
            if 'K' in df.columns and 'D' in df.columns and 'J' in df.columns:
                K = df['K'].iloc[i] if pd.notna(df['K'].iloc[i]) else 50
                D = df['D'].iloc[i] if pd.notna(df['D'].iloc[i]) else 50
                J = df['J'].iloc[i] if pd.notna(df['J'].iloc[i]) else 50
                
                if K > D and J > 20:  # KDJ金叉且J值不过低
                    signal_strength += 35
                elif K > 50:
                    signal_strength += 20
                    
            # 填坑确认：近期有回调再反弹
            if i >= 5:
                recent_prices = df['close'].iloc[i-4:i+1]
                if recent_prices.min() < recent_prices.iloc[0] and recent_prices.iloc[-1] > recent_prices.iloc[-2]:
                    signal_strength += 25
                    
            signals.append(min(100, signal_strength))
        
        return pd.Series(signals, index=df.index)
    
    def _calculate_combined_strategy_signal(self, df: pd.DataFrame) -> pd.Series:
        """计算4策略综合信号"""
        
        # 获取各策略信号
        bbi_kdj = self._calculate_bbi_kdj_signal(df)
        bbi_short_long = self._calculate_bbi_short_long_signal(df)
        breakout_volume_kdj = self._calculate_breakout_volume_kdj_signal(df)
        peak_kdj = self._calculate_peak_kdj_signal(df)
        
        # 策略权重配置
        weights = {
            'bbi_kdj': 0.30,           # 少负战法
            'bbi_short_long': 0.25,    # 补票战法  
            'breakout_volume_kdj': 0.25, # TePu战法
            'peak_kdj': 0.20           # 填坑战法
        }
        
        # 综合信号
        combined_signal = (
            bbi_kdj * weights['bbi_kdj'] +
            bbi_short_long * weights['bbi_short_long'] +
            breakout_volume_kdj * weights['breakout_volume_kdj'] +
            peak_kdj * weights['peak_kdj']
        )
        
        return combined_signal
    
    def create_unified_factor_dataset(self, 
                                    stock_codes: List[str],
                                    start_date: str,
                                    end_date: str,
                                    include_v2: bool = True,
                                    include_v4: bool = True,  # 🆕 支持V4因子
                                    include_strategies: bool = True) -> pd.DataFrame:
        """创建统一的因子数据集"""
        
        self.logger.info(f"为 {len(stock_codes)} 只股票创建统一因子数据集...")
        
        all_factors_data = []
        
        for stock_code in stock_codes:
            try:
                # 获取基础数据
                with sqlite3.connect(self.db_path) as conn:
                    query = """
                        SELECT 
                            dq.trade_date,
                            dq.open,
                            dq.high,
                            dq.low,
                            dq.close,
                            dq.volume,
                            ti.bbi,
                            ti.rsi6 as rsi,
                            ti.macd_dif,
                            ti.macd_dea,
                            ti.kdj_k as K,
                            ti.kdj_d as D, 
                            ti.kdj_j as J
                        FROM daily_quotes dq
                        JOIN securities s ON dq.security_id = s.id
                        LEFT JOIN technical_indicators ti ON ti.security_id = s.id 
                            AND ti.trade_date = dq.trade_date
                        WHERE s.code = ? 
                        AND dq.trade_date BETWEEN ? AND ?
                        ORDER BY dq.trade_date
                    """
                    
                    stock_df = pd.read_sql_query(
                        query, conn, params=[stock_code, start_date, end_date]
                    )
                
                if stock_df.empty:
                    continue
                
                # 创建结果DataFrame
                result_df = stock_df[['trade_date']].copy()
                result_df['stock_code'] = stock_code
                
                # 添加V2因子
                if include_v2:
                    v2_data = self.v2_extractor.extract_factors_for_stock(
                        stock_code, start_date, end_date
                    )
                    if not v2_data.empty:
                        # 合并V2因子
                        v2_factors = [col for col in v2_data.columns if col.startswith('v2_')]
                        merge_df = v2_data[['trade_date'] + v2_factors]
                        result_df = pd.merge(result_df, merge_df, on='trade_date', how='left')
                
                # 🆕 添加V4因子
                if include_v4:
                    v4_data = self.v4_extractor.extract_factors_for_stock(
                        stock_code, start_date, end_date
                    )
                    if not v4_data.empty:
                        # 合并V4因子
                        v4_factors = [col for col in v4_data.columns if col.startswith('v4_')]
                        merge_df = v4_data[['trade_date'] + v4_factors]
                        result_df = pd.merge(result_df, merge_df, on='trade_date', how='left')
                
                # 添加策略因子
                if include_strategies:
                    result_df['bbi_kdj_signal'] = self._calculate_bbi_kdj_signal(stock_df)
                    result_df['bbi_short_long_signal'] = self._calculate_bbi_short_long_signal(stock_df)
                    result_df['breakout_volume_kdj_signal'] = self._calculate_breakout_volume_kdj_signal(stock_df)
                    result_df['peak_kdj_signal'] = self._calculate_peak_kdj_signal(stock_df)
                    result_df['combined_strategy_signal'] = self._calculate_combined_strategy_signal(stock_df)
                
                # 添加基础技术指标
                result_df['close'] = stock_df['close']
                result_df['volume'] = stock_df['volume']
                result_df['bbi'] = stock_df['bbi']
                result_df['rsi'] = stock_df['rsi']
                
                all_factors_data.append(result_df)
                
            except Exception as e:
                self.logger.error(f"处理股票 {stock_code} 失败: {e}")
                continue
        
        if all_factors_data:
            final_df = pd.concat(all_factors_data, ignore_index=True)
            self.logger.info(f"成功创建包含 {len(final_df)} 条记录的统一因子数据集")
            return final_df
        else:
            return pd.DataFrame()
    
    def run_stock_selection(self, 
                           stock_pool: List[str],
                           trade_date: str,
                           strategy: str = "combined",
                           top_n: int = 50) -> Dict:
        """运行选股分析"""
        
        self.logger.info(f"运行 {strategy} 策略选股，候选池: {len(stock_pool)} 只股票")
        
        # 获取当日因子数据 - 针对V4策略只提取V4因子
        if strategy.startswith('v4_'):
            factor_data = self.create_unified_factor_dataset(
                stock_pool, trade_date, trade_date,
                include_v2=False,  # V4策略不需要V2因子
                include_v4=True,
                include_strategies=False  # V4策略不需要传统策略因子
            )
        else:
            factor_data = self.create_unified_factor_dataset(
                stock_pool, trade_date, trade_date
            )
        
        if factor_data.empty:
            return {
                'trade_date': trade_date,
                'strategy': strategy,
                'selected_stocks': [],
                'statistics': {}
            }
        
        # 根据策略选择评分列
        if strategy == "combined":
            score_column = "combined_strategy_signal"
        elif strategy == "v2_composite":
            score_column = "v2_composite_score"
        elif strategy == "v4_comprehensive":  # 🆕 V4综合评分
            score_column = "v4_comprehensive_score"
        elif strategy == "v4_squeeze_momentum":  # 🆕 V4挤压动量策略
            score_column = "v4_squeeze_release"  # 使用挤压释放因子作为主要信号
        elif strategy == "bbi_kdj":
            score_column = "bbi_kdj_signal"
        elif strategy == "breakout_volume":
            score_column = "breakout_volume_kdj_signal"
        else:
            score_column = "combined_strategy_signal"
        
        # 筛选有效数据
        valid_data = factor_data[factor_data[score_column].notna()].copy()
        
        if valid_data.empty:
            return {
                'trade_date': trade_date,
                'strategy': strategy,
                'selected_stocks': [],
                'statistics': {}
            }
        
        # 排序选出Top股票
        top_stocks = valid_data.nlargest(top_n, score_column)
        
        # 统计信息
        statistics = {
            'total_candidates': len(valid_data),
            'avg_score': valid_data[score_column].mean(),
            'max_score': valid_data[score_column].max(),
            'min_score': valid_data[score_column].min(),
            'score_distribution': {
                '90-100': len(valid_data[valid_data[score_column] >= 90]),
                '80-90': len(valid_data[(valid_data[score_column] >= 80) & (valid_data[score_column] < 90)]),
                '70-80': len(valid_data[(valid_data[score_column] >= 70) & (valid_data[score_column] < 80)]),
                '60-70': len(valid_data[(valid_data[score_column] >= 60) & (valid_data[score_column] < 70)]),
                '<60': len(valid_data[valid_data[score_column] < 60])
            }
        }
        
        # 格式化结果
        selected_stocks = []
        for _, row in top_stocks.iterrows():
            selected_stocks.append({
                'stock_code': row['stock_code'],
                'score': round(row[score_column], 1),
                'rank': len(selected_stocks) + 1
            })
        
        return {
            'trade_date': trade_date,
            'strategy': strategy,
            'selected_stocks': selected_stocks,
            'statistics': statistics
        }
    
    def generate_selection_report(self, 
                                selection_result: Dict,
                                output_path: Optional[str] = None) -> str:
        """生成选股报告"""
        
        if output_path is None:
            date_str = selection_result['trade_date'].replace('-', '')
            strategy = selection_result['strategy']
            output_path = f"reports/factor_management/因子管理选股报告_{strategy}_{date_str}.md"
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告内容
        report_content = self._format_selection_report(selection_result)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"选股报告已生成: {output_path}")
        return output_path
    
    def _format_selection_report(self, result: Dict) -> str:
        """格式化选股报告"""
        
        trade_date = result['trade_date']
        strategy = result['strategy']
        selected_stocks = result['selected_stocks']
        statistics = result['statistics']
        
        strategy_names = {
            'combined': '4策略综合',
            'v2_composite': 'V2综合评分',
            'v4_comprehensive': 'V4挤压动量综合评分',  # 🆕 V4策略名称
            'v4_squeeze_momentum': 'V4挤压动量策略',   # 🆕 V4挤压动量策略
            'bbi_kdj': '少负战法(BBI+KDJ)',
            'breakout_volume': 'TePu战法(量价突破)'
        }
        
        strategy_name = strategy_names.get(strategy, strategy)
        
        report = f"""# 因子管理框架选股报告

**日期**: {trade_date}  
**策略**: {strategy_name}  
**选出股票**: {len(selected_stocks)}只  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 选股统计

- **候选股票总数**: {statistics.get('total_candidates', 0)}只
- **平均评分**: {statistics.get('avg_score', 0):.1f}分
- **最高评分**: {statistics.get('max_score', 0):.1f}分
- **最低评分**: {statistics.get('min_score', 0):.1f}分

### 评分分布

"""
        
        score_dist = statistics.get('score_distribution', {})
        for range_name, count in score_dist.items():
            percentage = (count / statistics.get('total_candidates', 1)) * 100
            report += f"- **{range_name}分**: {count}只 ({percentage:.1f}%)\n"
        
        report += f"\n## 🎯 精选股票 (Top {len(selected_stocks)})\n\n"
        report += "| 排名 | 股票代码 | 评分 | 策略信号强度 |\n"
        report += "|------|----------|------|-------------|\n"
        
        for stock in selected_stocks:
            signal_strength = "强" if stock['score'] >= 80 else "中" if stock['score'] >= 60 else "弱"
            report += f"| {stock['rank']} | {stock['stock_code']} | {stock['score']}分 | {signal_strength} |\n"
        
        report += f"""

## 🔧 技术说明

### 因子管理框架优势

1. **统一因子计算**: 所有因子预计算存储，查询效率高
2. **版本管理**: 支持多版本因子并存，便于对比测试
3. **标准化接口**: 统一的因子获取和计算接口
4. **扩展性强**: 轻松添加新因子和策略

### {strategy_name} 策略说明

"""
        
        if strategy == "combined":
            report += """
该策略综合4个经典选股策略的信号：
- **少负战法(30%)**: BBI+KDJ组合信号
- **补票战法(25%)**: BBI短长期RSV对比
- **TePu战法(25%)**: 量价突破+KDJ确认
- **填坑战法(20%)**: 峰值检测+KDJ回升
"""
        elif strategy == "v2_composite":
            report += """
基于3949只股票实际表现优化的V2评分系统：
- **动量因子(40%)**: 价格+成交量+技术+趋势动量
- **均值回归(25%)**: 价格修复能力评估
- **量价突破(20%)**: 突破确认信号
- **相对强度(10%)**: 相对市场表现
- **稳定性(5%)**: 波动率风险控制
"""
        elif strategy == "v4_comprehensive":
            report += """
V4挤压动量增强评分系统，集成John Carter的TTM Squeeze指标：
- **技术指标(50%)**: KDJ强度15% + RSI动量14% + BBI趋势10% + 成交量异动11%
- **🆕挤压动量(20%)**: 挤压状态5% + 挤压释放6% + 动量方向5% + 动量加速度4%
- **基本面(8%)**: PE估值2% + PB估值2% + 市值因子2% + 换手率活跃度2%
- **市场表现(18%)**: 价格动量13% + 相对强度3% + 波动率风险2%
- **市场环境(4%)**: 市场贝塔1% + 板块轮动1.5% + 流动性1.5%

🔥 **挤压动量核心逻辑**：
- 识别布林带在肯特纳通道内收窄的"挤压"状态
- 捕捉从低波动到高波动的转换点
- 利用线性回归动量确认突破方向
"""
        elif strategy == "v4_squeeze_momentum":
            report += """
V4挤压动量专项策略，专注于挤压释放信号：
- **核心信号**: 挤压状态突然释放的瞬间
- **技术原理**: 布林带突破肯特纳通道边界
- **适用场景**: 横盘整理后的方向性突破
- **风险控制**: 结合动量方向和加速度确认

⚡ **策略优势**：
- 提前识别价格突破的酝酿期
- 降低假突破的干扰
- 捕捉波动率扩张的黄金时机
"""
        
        report += f"""

## ⚠️ 风险提示

1. **历史数据基础**: 因子效果基于历史数据，未来可能变化
2. **市场环境影响**: 不同市场环境下因子有效性可能不同
3. **组合投资**: 建议分散投资，控制单股仓位
4. **止损策略**: 建议设置合适的止损点
5. **持续跟踪**: 定期评估因子表现，及时调整策略

---

*本报告由因子管理框架生成，仅供参考，不构成投资建议*
"""
        
        return report


def main():
    """测试适配器功能"""
    
    adapter = StockSelectorAdapter()
    
    # 测试创建统一因子数据集
    print("测试创建统一因子数据集...")
    test_stocks = ['000001', '000002', '000858']
    factor_data = adapter.create_unified_factor_dataset(
        test_stocks, '2025-08-15', '2025-08-18'
    )
    print(f"创建了 {len(factor_data)} 条因子记录")
    print("因子列表:", factor_data.columns.tolist())
    
    # 测试选股功能
    print("\n测试选股功能...")
    selection_result = adapter.run_stock_selection(
        test_stocks, '2025-08-18', strategy="combined", top_n=3
    )
    print(f"选股结果: {len(selection_result['selected_stocks'])} 只股票")
    for stock in selection_result['selected_stocks']:
        print(f"  {stock['stock_code']}: {stock['score']}分")
    
    # 生成报告
    print("\n生成选股报告...")
    report_path = adapter.generate_selection_report(selection_result)
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    main()