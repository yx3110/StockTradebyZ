#!/usr/bin/env python3
"""
基于现有数据创建增强版证券列表
从现有的股票数据文件推断基本信息，并添加常见的行业信息
"""

import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("securities_enhancer")

def get_industry_by_code(stock_code: str, stock_name: str) -> str:
    """根据股票代码和名称推断行业"""
    # 常见行业关键词映射
    industry_keywords = {
        '银行': ['银行', '农商', '农信', '信用社'],
        '医药生物': ['医药', '生物', '制药', '医疗', '健康', '康复', '药业'],
        '电子': ['电子', '科技', '芯片', '半导体', '集成', '通信'],
        '房地产': ['地产', '置业', '房产', '建设', '城建'],
        '汽车': ['汽车', '车辆', '客车', '货车', '汽配'],
        '钢铁': ['钢铁', '钢构', '冶金', '钢管'],
        '化工': ['化工', '化学', '石化', '塑料', '橡胶'],
        '机械设备': ['机械', '设备', '重工', '装备', '工程'],
        '电力设备': ['电力', '电气', '变压', '输配电'],
        '食品饮料': ['食品', '饮料', '乳业', '酒业', '茶业'],
        '纺织服装': ['纺织', '服装', '服饰', '纤维'],
        '交通运输': ['运输', '航空', '港口', '物流', '快递'],
        '公用事业': ['水务', '燃气', '供水', '环保'],
        '建筑材料': ['建材', '水泥', '玻璃', '陶瓷'],
        '采掘': ['煤炭', '石油', '天然气', '采矿'],
        '有色金属': ['有色', '铝业', '铜业', '黄金'],
        '传媒': ['传媒', '广告', '影视', '出版'],
        '计算机': ['软件', '信息', '数据', '云计算', '互联网'],
        '军工': ['军工', '航天', '航空', '船舶'],
        '农林牧渔': ['农业', '林业', '牧业', '渔业', '种业'],
    }
    
    # ETF和基金处理
    if stock_code.startswith(('50', '51', '15', '16')):
        if 'ETF' in stock_name:
            return 'ETF基金'
        else:
            return '其他基金'
    
    # 根据股票名称匹配行业
    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            if keyword in stock_name:
                return industry
    
    return '综合'

def get_area_by_code(stock_code: str, stock_name: str) -> str:
    """根据股票代码和名称推断注册地"""
    # 地区关键词
    area_keywords = {
        '北京': ['北京', '京东', '京能', '京投'],
        '上海': ['上海', '上港', '上汽', '上实'],
        '深圳': ['深圳', '深振业', '深物业', '深赛格'],
        '广东': ['广东', '广州', '佛山', '东莞', '粤'],
        '浙江': ['浙江', '杭州', '宁波', '温州', '浙能'],
        '江苏': ['江苏', '南京', '苏州', '无锡', '苏宁'],
        '山东': ['山东', '青岛', '济南', '烟台', '鲁'],
        '四川': ['四川', '成都', '川', '蜀'],
        '湖北': ['湖北', '武汉', '鄂'],
        '湖南': ['湖南', '长沙', '湘'],
        '河南': ['河南', '郑州', '豫'],
        '河北': ['河北', '石家庄', '冀'],
        '安徽': ['安徽', '合肥', '皖'],
        '福建': ['福建', '福州', '厦门', '闽'],
        '江西': ['江西', '南昌', '赣'],
        '辽宁': ['辽宁', '沈阳', '大连', '辽'],
        '黑龙江': ['黑龙江', '哈尔滨', '黑'],
        '吉林': ['吉林', '长春', '吉'],
        '山西': ['山西', '太原', '晋'],
        '陕西': ['陕西', '西安', '陕'],
        '甘肃': ['甘肃', '兰州', '甘'],
        '青海': ['青海', '西宁', '青'],
        '宁夏': ['宁夏', '银川', '宁'],
        '新疆': ['新疆', '乌鲁木齐', '新'],
        '西藏': ['西藏', '拉萨', '藏'],
        '内蒙古': ['内蒙', '呼和浩特', '蒙'],
        '广西': ['广西', '南宁', '桂'],
        '海南': ['海南', '海口', '琼'],
        '重庆': ['重庆', '渝'],
        '天津': ['天津', '津'],
    }
    
    # 根据股票名称匹配地区
    for area, keywords in area_keywords.items():
        for keyword in keywords:
            if keyword in stock_name:
                return area
    
    # 根据代码段推断（粗略）
    if stock_code.startswith('60'):
        return '上海'
    elif stock_code.startswith(('00', '30')):
        return '深圳'
    elif stock_code.startswith('688'):
        return '上海'
    
    return '未知'

def create_enhanced_securities_list():
    """创建增强版证券列表"""
    data_dir = Path("../full_securities_data")
    
    # 读取原始证券列表
    original_file = data_dir / "securities_list.csv"
    if not original_file.exists():
        logger.error(f"原始证券列表文件不存在: {original_file}")
        return False
    
    logger.info("读取原始证券列表...")
    df = pd.read_csv(original_file, dtype={'code': str})
    
    logger.info(f"处理 {len(df)} 只证券的基本面信息...")
    
    # 添加行业和地区信息
    df['industry'] = df.apply(lambda row: get_industry_by_code(row['code'], row['name']), axis=1)
    df['area'] = df.apply(lambda row: get_area_by_code(row['code'], row['name']), axis=1)
    
    # 添加上市日期（设为默认值，因为无法准确推断）
    df['list_date'] = '19900101'  # 默认上市日期
    
    # 重新排列列
    df = df[['ts_code', 'code', 'name', 'type', 'market', 'industry', 'area', 'list_date']]
    
    # 备份原文件
    if original_file.exists():
        backup_file = data_dir / "securities_list_original.csv"
        original_file.rename(backup_file)
        logger.info(f"原文件已备份为: {backup_file}")
    
    # 保存增强版文件
    df.to_csv(original_file, index=False, encoding='utf-8')
    
    logger.info(f"成功创建增强版证券列表: {original_file}")
    logger.info("新增字段: industry(行业), area(地区), list_date(上市日期)")
    
    # 显示行业分布统计
    logger.info("\n行业分布统计:")
    industry_stats = df['industry'].value_counts()
    print(industry_stats.head(10))
    
    # 显示地区分布统计
    logger.info("\n地区分布统计:")
    area_stats = df['area'].value_counts()
    print(area_stats.head(10))
    
    # 显示样例数据
    logger.info("\n增强后的数据样例:")
    print(df.head(10).to_string(index=False))
    
    return True

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("创建增强版证券列表")
    logger.info("=" * 60)
    
    success = create_enhanced_securities_list()
    
    if success:
        logger.info("增强版证券列表创建完成！")
    else:
        logger.error("增强版证券列表创建失败！")

if __name__ == "__main__":
    main()