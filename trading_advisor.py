#!/usr/bin/env python3
"""
交易建议生成器
基于当前持仓和选股报告生成智能交易建议
整合ChatGPT-Micro-Cap-Experiment的交易逻辑和风险管理
"""

import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class TradingAdvisor:
    """交易建议生成器"""
    
    def __init__(self, config_path: str = "config.json", db_path: str = "data_adapter/stock_data.db"):
        """初始化交易建议器"""
        self.config_path = config_path
        self.db_path = db_path
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.load_config()
        
        # 风险管理参数 (来自ChatGPT实验)
        self.max_position_size = 0.1  # 单个股票最大仓位10%
        self.stop_loss_ratio = 0.08   # 止损比例8%
        self.max_portfolio_stocks = 10  # 最大持股数量
        self.cash_reserve_ratio = 0.05  # 现金储备比例5%
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 不存在，使用默认配置")
            self.config = {}
    
    def load_current_positions(self) -> pd.DataFrame:
        """加载当前持仓数据"""
        positions_file = "daily_positions.md"
        
        if not os.path.exists(positions_file):
            print(f"持仓文件 {positions_file} 不存在")
            return pd.DataFrame()
        
        # 解析markdown格式的持仓文件
        with open(positions_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找最新日期的持仓表格
        lines = content.split('\n')
        positions = []
        in_table = False
        
        for line in lines:
            if '## 2025-08-04' in line:  # 查找最新日期
                in_table = True
                continue
            elif line.startswith('## ') and in_table:
                break
            elif in_table and '|' in line and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3 and parts[0] != '股票代码':
                    try:
                        positions.append({
                            'stock_code': parts[0],
                            'stock_name': parts[1],
                            'shares': int(parts[2]) if parts[2] != '0' else 0
                        })
                    except (ValueError, IndexError):
                        continue
        
        return pd.DataFrame(positions)
    
    def load_latest_report(self) -> Optional[pd.DataFrame]:
        """加载最新的选股报告"""
        reports_dir = "reports/daily_selection"
        if not os.path.exists(reports_dir):
            print("未找到选股报告目录")
            return None
        
        # 查找最新的报告文件
        report_files = [f for f in os.listdir(reports_dir) if f.startswith('选股分析报告_')]
        if not report_files:
            print("未找到选股报告")
            return None
        
        latest_file = sorted(report_files)[-1]
        return self.parse_report(os.path.join(reports_dir, latest_file))
    
    def parse_report(self, report_path: str) -> pd.DataFrame:
        """解析选股报告"""
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        recommendations = []
        lines = content.split('\n')
        current_stock = {}
        
        for line in lines:
            if line.startswith('### ') and ('- ' in line):  # 股票标题行
                if current_stock:
                    recommendations.append(current_stock)
                    current_stock = {}
                
                # 提取股票代码和名称
                parts = line.replace('### ', '').split(' - ')
                if len(parts) >= 2:
                    # 提取股票代码 (去掉序号)
                    code_part = parts[0].strip()
                    if '. ' in code_part:
                        code_part = code_part.split('. ')[1]
                    current_stock['stock_code'] = code_part
                    current_stock['stock_name'] = parts[1].strip()
            
            elif '- **收盘价**:' in line:
                try:
                    price = float(line.split('**收盘价**:')[1].split('元')[0].strip())
                    current_stock['current_price'] = price
                except:
                    pass
            
            elif '- **建议买入价**:' in line:
                try:
                    price = float(line.split('**建议买入价**:')[1].split('元')[0].strip())
                    current_stock['suggested_buy_price'] = price
                except:
                    pass
            
            elif '- **建议止损价**:' in line:
                try:
                    price = float(line.split('**建议止损价**:')[1].split('元')[0].strip())
                    current_stock['stop_loss_price'] = price
                except:
                    pass
            
            elif '- **建议止盈价**:' in line:
                try:
                    price = float(line.split('**建议止盈价**:')[1].split('元')[0].strip())
                    current_stock['target_price'] = price
                except:
                    pass
            
            elif '- **综合评分**:' in line:
                try:
                    score = float(line.split('**综合评分**:')[1].split('分')[0].strip())
                    current_stock['score'] = score
                except:
                    pass
            
            elif '- **通过策略数**:' in line:
                try:
                    count = int(line.split('**通过策略数**:')[1].split('个')[0].strip())
                    current_stock['strategy_count'] = count
                except:
                    pass
        
        if current_stock:
            recommendations.append(current_stock)
        
        return pd.DataFrame(recommendations)
    
    def get_stock_price(self, stock_code: str) -> Tuple[float, str]:
        """从SQLite数据库获取股票最新价格和名称"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT close, name, trade_date 
            FROM latest_quotes 
            WHERE code = ?
            """
            
            cursor.execute(query, (stock_code,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                price, name, trade_date = result
                return float(price), name
            else:
                print(f"警告: 未找到股票 {stock_code} 的价格数据")
                return 0.0, ""
                
        except Exception as e:
            print(f"获取股票 {stock_code} 价格时出错: {e}")
            return 0.0, ""
    
    def calculate_portfolio_value(self, positions: pd.DataFrame) -> Tuple[float, Dict]:
        """计算当前投资组合价值"""
        total_value = 0.0
        position_details = {}
        
        for _, pos in positions.iterrows():
            if pos['shares'] > 0:
                stock_code = pos['stock_code']
                shares = pos['shares']
                
                # 从数据库获取最新价格
                current_price, stock_name = self.get_stock_price(stock_code)
                
                if current_price > 0:
                    position_value = shares * current_price
                    total_value += position_value
                    
                    position_details[stock_code] = {
                        'name': stock_name,
                        'shares': shares,
                        'price': current_price,
                        'value': position_value
                    }
                else:
                    print(f"警告: 股票 {stock_code} 价格为0，可能数据缺失")
        
        return total_value, position_details
    
    def generate_sell_recommendations(self, positions: pd.DataFrame, recommendations: pd.DataFrame) -> List[Dict]:
        """生成卖出建议"""
        sell_recommendations = []
        
        for _, pos in positions.iterrows():
            if pos['shares'] == 0:  # 跳过已清仓的股票
                continue
                
            stock_code = pos['stock_code']
            stock_name = pos['stock_name']
            shares = pos['shares']
            
            # 从数据库获取真实当前价格
            current_price, db_stock_name = self.get_stock_price(stock_code)
            if db_stock_name:  # 如果数据库有名称，使用数据库的名称
                stock_name = db_stock_name
            
            # 在推荐列表中查找该股票
            rec = recommendations[recommendations['stock_code'] == stock_code]
            
            if rec.empty:
                # 不在推荐列表中的股票 - 建议考虑卖出
                sell_recommendations.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'shares': shares,
                    'current_price': current_price,
                    'action': 'CONSIDER_SELL',
                    'reason': '该股票不在最新推荐列表中，建议考虑减仓或清仓',
                    'priority': 'MEDIUM',
                    'confidence': 0.6
                })
            else:
                rec_data = rec.iloc[0]
                stop_loss_price = rec_data.get('stop_loss_price', 0)
                target_price = rec_data.get('target_price', 0)
                score = rec_data.get('score', 0)
                
                # 检查止损条件
                if current_price > 0 and stop_loss_price > 0:
                    if current_price <= stop_loss_price * 1.02:  # 2%容错
                        sell_recommendations.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'shares': shares,
                            'current_price': current_price,
                            'action': 'SELL_STOP_LOSS',
                            'reason': f'触发止损价位，当前价{current_price:.2f}元接近止损价{stop_loss_price:.2f}元',
                            'priority': 'HIGH',
                            'confidence': 0.9,
                            'suggested_price': current_price
                        })
                
                # 检查止盈条件
                elif current_price > 0 and target_price > 0:
                    if current_price >= target_price * 0.95:  # 接近目标价
                        sell_recommendations.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'shares': shares,
                            'current_price': current_price,
                            'action': 'CONSIDER_TAKE_PROFIT',
                            'reason': f'接近目标价位，当前价{current_price:.2f}元接近目标价{target_price:.2f}元',
                            'priority': 'LOW',
                            'confidence': 0.7,
                            'suggested_price': current_price
                        })
                
                # 评分较低的股票
                elif score > 0 and score < 60:
                    sell_recommendations.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'shares': shares,
                        'current_price': current_price,
                        'action': 'CONSIDER_SELL',
                        'reason': f'综合评分较低({score:.1f}分)，建议考虑减仓',
                        'priority': 'MEDIUM',
                        'confidence': 0.6
                    })
        
        return sell_recommendations
    
    def generate_buy_recommendations(self, positions: pd.DataFrame, recommendations: pd.DataFrame, 
                                   total_portfolio_value: float) -> List[Dict]:
        """生成买入建议"""
        buy_recommendations = []
        
        # 当前持仓的股票代码
        current_positions = set(positions[positions['shares'] > 0]['stock_code'])
        
        # 筛选高质量推荐股票
        high_quality_recs = recommendations[
            (recommendations['score'] >= 75) &  # 高评分
            (recommendations['strategy_count'] >= 2)  # 多策略验证
        ].copy()
        
        # 按评分排序
        high_quality_recs = high_quality_recs.sort_values('score', ascending=False)
        
        position_count = len(current_positions)
        
        # 计算可用于新投资的资金 (假设可以调仓，释放部分资金)
        available_cash = total_portfolio_value * 0.3  # 假设30%的资金可用于调仓
        
        for _, rec in high_quality_recs.head(10).iterrows():  # 只考虑前10只
            stock_code = rec['stock_code']
            stock_name = rec['stock_name']
            
            # 跳过已持有的股票
            if stock_code in current_positions:
                continue
            
            # 检查投资组合是否已经太分散
            if position_count >= self.max_portfolio_stocks:
                break
                
            current_price = rec.get('current_price', 0)
            suggested_buy_price = rec.get('suggested_buy_price', current_price)
            stop_loss_price = rec.get('stop_loss_price', 0)
            target_price = rec.get('target_price', 0)
            score = rec.get('score', 0)
            
            if current_price <= 0:
                continue
            
            # 计算建议仓位大小
            max_investment = min(
                available_cash * 0.5,  # 单次投资不超过可用资金的50%
                total_portfolio_value * self.max_position_size  # 不超过总资产的10%
            )
            
            suggested_shares = int(max_investment / suggested_buy_price)
            
            if suggested_shares < 100:  # 最少100股
                continue
            
            # 计算风险收益比
            if stop_loss_price > 0 and target_price > 0:
                potential_loss = (suggested_buy_price - stop_loss_price) / suggested_buy_price
                potential_gain = (target_price - suggested_buy_price) / suggested_buy_price
                risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
            else:
                risk_reward_ratio = 0
            
            # 确定优先级
            if score >= 80 and risk_reward_ratio >= 3:
                priority = 'HIGH'
                confidence = 0.8
            elif score >= 75 and risk_reward_ratio >= 2:
                priority = 'MEDIUM'
                confidence = 0.7
            else:
                priority = 'LOW'
                confidence = 0.6
            
            buy_recommendations.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'action': 'BUY',
                'suggested_shares': suggested_shares,
                'suggested_price': suggested_buy_price,
                'current_price': current_price,
                'stop_loss_price': stop_loss_price,
                'target_price': target_price,
                'score': score,
                'risk_reward_ratio': risk_reward_ratio,
                'investment_amount': suggested_shares * suggested_buy_price,
                'priority': priority,
                'confidence': confidence,
                'reason': f'高质量推荐股票(评分{score:.1f}分，风险收益比{risk_reward_ratio:.1f}:1)'
            })
            
            available_cash -= suggested_shares * suggested_buy_price
            position_count += 1
        
        return buy_recommendations
    
    def generate_hold_recommendations(self, positions: pd.DataFrame, recommendations: pd.DataFrame) -> List[Dict]:
        """生成持有建议"""
        hold_recommendations = []
        
        for _, pos in positions.iterrows():
            if pos['shares'] == 0:
                continue
                
            stock_code = pos['stock_code']
            stock_name = pos['stock_name']
            shares = pos['shares']
            
            # 在推荐列表中查找该股票
            rec = recommendations[recommendations['stock_code'] == stock_code]
            
            if not rec.empty:
                rec_data = rec.iloc[0]
                current_price = rec_data.get('current_price', 0)
                stop_loss_price = rec_data.get('stop_loss_price', 0)
                target_price = rec_data.get('target_price', 0)
                score = rec_data.get('score', 0)
                
                # 如果股票仍然在推荐列表中且条件良好
                if (score >= 70 and 
                    current_price > stop_loss_price * 1.05 and  # 远离止损价
                    current_price < target_price * 0.9):  # 尚未达到目标价
                    
                    hold_recommendations.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'shares': shares,
                        'action': 'HOLD',
                        'current_price': current_price,
                        'stop_loss_price': stop_loss_price,
                        'target_price': target_price,
                        'score': score,
                        'reason': f'股票基本面良好(评分{score:.1f}分)，继续持有等待目标价位',
                        'priority': 'LOW',
                        'confidence': 0.8
                    })
        
        return hold_recommendations
    
    def generate_report(self, sell_recs: List[Dict], buy_recs: List[Dict], 
                       hold_recs: List[Dict], positions: pd.DataFrame) -> str:
        """生成交易建议报告"""
        report = f"""# 📊 每日交易建议报告

## 📋 报告概要
- **生成日期**: {self.today}
- **分析股票数**: {len(positions)}只
- **卖出建议**: {len(sell_recs)}条
- **买入建议**: {len(buy_recs)}条  
- **持有建议**: {len(hold_recs)}条

---

## 🔴 卖出建议 ({len(sell_recs)}条)

"""
        
        # 按优先级排序卖出建议
        sell_recs_sorted = sorted(sell_recs, key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], reverse=True)
        
        for i, rec in enumerate(sell_recs_sorted, 1):
            report += f"""### {i}. {rec['stock_code']} - {rec['stock_name']}

- **操作**: {rec['action']}
- **优先级**: {rec['priority']}
- **持有股数**: {rec['shares']}股
- **建议**: {rec['reason']}
- **信心指数**: {rec['confidence']:.1%}
"""
            if 'suggested_price' in rec:
                report += f"- **建议价格**: {rec['suggested_price']:.2f}元\n"
            report += "\n---\n\n"
        
        if not sell_recs:
            report += "暂无卖出建议\n\n---\n\n"
        
        report += f"""## 🟢 买入建议 ({len(buy_recs)}条)

"""
        
        # 按优先级排序买入建议
        buy_recs_sorted = sorted(buy_recs, key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], reverse=True)
        
        for i, rec in enumerate(buy_recs_sorted, 1):
            report += f"""### {i}. {rec['stock_code']} - {rec['stock_name']}

- **操作**: 买入
- **优先级**: {rec['priority']}
- **建议股数**: {rec['suggested_shares']}股
- **建议买入价**: {rec['suggested_price']:.2f}元
- **当前价格**: {rec['current_price']:.2f}元
- **止损价**: {rec['stop_loss_price']:.2f}元
- **目标价**: {rec['target_price']:.2f}元
- **投资金额**: {rec['investment_amount']:.0f}元
- **综合评分**: {rec['score']:.1f}分
- **风险收益比**: {rec['risk_reward_ratio']:.1f}:1
- **建议理由**: {rec['reason']}
- **信心指数**: {rec['confidence']:.1%}

---

"""
        
        if not buy_recs:
            report += "暂无买入建议\n\n---\n\n"
        
        report += f"""## 🔵 持有建议 ({len(hold_recs)}条)

"""
        
        for i, rec in enumerate(hold_recs, 1):
            report += f"""### {i}. {rec['stock_code']} - {rec['stock_name']}

- **操作**: 继续持有
- **持有股数**: {rec['shares']}股
- **当前价格**: {rec['current_price']:.2f}元
- **止损价**: {rec['stop_loss_price']:.2f}元
- **目标价**: {rec['target_price']:.2f}元
- **综合评分**: {rec['score']:.1f}分
- **持有理由**: {rec['reason']}
- **信心指数**: {rec['confidence']:.1%}

---

"""
        
        if not hold_recs:
            report += "暂无持有建议\n\n---\n\n"
        
        report += f"""## ⚠️ 风险提示

### 交易原则
1. **止损纪律**: 严格执行止损，单笔亏损不超过总资金的3%
2. **仓位控制**: 单只股票仓位不超过总资金的10%
3. **分散投资**: 持股数量控制在{self.max_portfolio_stocks}只以内
4. **现金储备**: 保留{self.cash_reserve_ratio:.1%}现金用于机会把握

### 免责声明
本报告基于量化分析生成，仅供参考，不构成投资建议。
投资有风险，决策需谨慎。

---

📊 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Generated by TradingAdvisor
"""
        
        return report
    
    def run(self) -> str:
        """运行交易建议生成器"""
        print("🚀 启动交易建议生成器...")
        
        # 加载数据
        positions = self.load_current_positions()
        recommendations = self.load_latest_report()
        
        if positions.empty:
            return "❌ 未找到持仓数据，无法生成建议"
        
        if recommendations is None or recommendations.empty:
            return "❌ 未找到选股报告，无法生成建议"
        
        print(f"📊 加载了 {len(positions)} 只持仓股票")
        print(f"📋 加载了 {len(recommendations)} 只推荐股票")
        
        # 计算当前投资组合总价值
        total_portfolio_value, position_details = self.calculate_portfolio_value(positions)
        print(f"💰 当前投资组合总价值: {total_portfolio_value:.2f}元")
        
        # 显示持仓详情
        print("📋 持仓详情:")
        for code, details in position_details.items():
            print(f"  {code} {details['name']}: {details['shares']}股 × {details['price']:.2f}元 = {details['value']:.2f}元")
        
        # 生成建议
        sell_recs = self.generate_sell_recommendations(positions, recommendations)
        buy_recs = self.generate_buy_recommendations(positions, recommendations, total_portfolio_value)
        hold_recs = self.generate_hold_recommendations(positions, recommendations)
        
        # 生成报告
        report = self.generate_report(sell_recs, buy_recs, hold_recs, positions)
        
        # 保存报告到reports文件夹
        reports_dir = "reports/trading_advice"
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"{reports_dir}/交易建议报告_{self.today.replace('-', '')}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 交易建议报告已生成: {report_filename}")
        return report


def main():
    """主函数"""
    advisor = TradingAdvisor()
    
    # 生成建议 (基于当前持仓价值)
    report = advisor.run()
    
    # 显示摘要
    print("\n" + "="*60)
    print("📊 交易建议摘要")
    print("="*60)
    
    lines = report.split('\n')
    for line in lines:
        if line.startswith('- **') and ('建议' in line or '股票数' in line):
            print(line)
    
    print("\n详细建议请查看生成的报告文件。")


if __name__ == "__main__":
    main()