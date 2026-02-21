"""
fetch_data - 数据抓取模块

核心脚本:
- quick_daily_update.py            每日数据更新管道（生产入口）
- technical_indicator_calculator.py 技术指标批量计算
- v39_feature_cache_updater.py     ML特征缓存（v3.9/v3.95）
- data_quality_check_db.py         数据质量验证
- v39_data_backfill.py             v39数据初始化/回填
- backfill_historical_data.py      历史数据修补（daily_basic NULL / 指数数据）
"""
