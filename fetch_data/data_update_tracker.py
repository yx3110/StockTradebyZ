#!/usr/bin/env python3
"""
数据更新状态追踪系统
功能：
1. 记录每次数据更新的时间和详情
2. 生成数据更新状态报告
3. 检查数据最新性
4. 创建更新历史日志
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("data_tracker")

class DataUpdateTracker:
    """数据更新追踪器"""
    
    def __init__(self, data_dir: str = "full_securities_data"):
        self.data_dir = Path(data_dir)
        self.metadata_file = self.data_dir / "update_metadata.json"
        self.update_log_file = self.data_dir / "update_history.log"
        self.status_file = Path("data_update_status.json")
        
    def record_update(self, start_date: str, end_date: str, 
                     success_count: int, total_count: int,
                     failed_files: List[str] = None):
        """记录数据更新信息"""
        update_info = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_start_date": start_date,
            "data_end_date": end_date,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": f"{success_count/total_count*100:.1f}%",
            "failed_files": failed_files or []
        }
        
        # 保存到元数据文件
        self._save_metadata(update_info)
        
        # 追加到历史日志
        self._append_to_log(update_info)
        
        # 更新状态文件
        self._update_status(update_info)
        
        logger.info(f"数据更新记录已保存: {end_date}")
        
    def _save_metadata(self, update_info: dict):
        """保存元数据到JSON文件"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(update_info, f, ensure_ascii=False, indent=2)
            
    def _append_to_log(self, update_info: dict):
        """追加到历史日志"""
        with open(self.update_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{json.dumps(update_info, ensure_ascii=False)}\n")
            
    def _update_status(self, update_info: dict):
        """更新总体状态文件"""
        status = {
            "last_update": update_info["update_time"],
            "latest_data_date": update_info["data_end_date"],
            "data_range": f"{update_info['data_start_date']} - {update_info['data_end_date']}",
            "total_files": update_info["total_count"],
            "success_rate": update_info["success_rate"],
            "status": "up_to_date" if update_info["success_rate"].startswith("100") else "partial_update"
        }
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
            
    def get_update_status(self) -> Dict:
        """获取当前更新状态"""
        if not self.metadata_file.exists():
            return {"status": "no_update_info", "message": "未找到更新信息"}
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 检查数据是否过时
        last_update = datetime.strptime(metadata["update_time"], "%Y-%m-%d %H:%M:%S")
        days_old = (datetime.now() - last_update).days
        
        metadata["days_since_update"] = days_old
        metadata["is_outdated"] = days_old > 1  # 超过1天算过时
        
        return metadata
        
    def check_data_freshness(self) -> Tuple[bool, Dict]:
        """检查数据新鲜度"""
        status = self.get_update_status()
        
        if status.get("status") == "no_update_info":
            return False, status
            
        is_fresh = not status.get("is_outdated", True)
        
        return is_fresh, status
        
    def scan_data_dates(self, sample_size: int = 10) -> Dict:
        """扫描数据文件获取日期范围"""
        if not self.data_dir.exists():
            return {"error": "数据目录不存在"}
            
        csv_files = list(self.data_dir.glob("*.csv"))[:sample_size]
        
        if not csv_files:
            return {"error": "未找到数据文件"}
            
        date_info = {
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sample_files": len(csv_files),
            "date_ranges": []
        }
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    date_info["date_ranges"].append({
                        "file": csv_file.name,
                        "min_date": df['date'].min().strftime("%Y-%m-%d"),
                        "max_date": df['date'].max().strftime("%Y-%m-%d"),
                        "rows": len(df)
                    })
            except Exception as e:
                logger.warning(f"扫描 {csv_file.name} 失败: {e}")
                
        # 计算整体日期范围
        if date_info["date_ranges"]:
            all_min_dates = [item["min_date"] for item in date_info["date_ranges"]]
            all_max_dates = [item["max_date"] for item in date_info["date_ranges"]]
            
            date_info["overall_min_date"] = min(all_min_dates)
            date_info["overall_max_date"] = max(all_max_dates)
            
        return date_info
        
    def generate_status_report(self) -> str:
        """生成数据状态报告"""
        # 获取元数据
        status = self.get_update_status()
        
        # 扫描实际数据
        scan_result = self.scan_data_dates()
        
        # 生成报告
        report = f"""# 📊 数据更新状态报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔄 最近更新信息
"""
        
        if status.get("status") == "no_update_info":
            report += "- ❌ 未找到更新记录\n"
        else:
            report += f"""- **更新时间**: {status['update_time']}
- **数据日期**: {status['data_end_date']}
- **覆盖范围**: {status['data_start_date']} 至 {status['data_end_date']}
- **更新文件**: {status['success_count']}/{status['total_count']}
- **成功率**: {status['success_rate']}
- **距今天数**: {status['days_since_update']}天
- **数据状态**: {'⚠️ 需要更新' if status['is_outdated'] else '✅ 最新'}
"""
        
        # 添加扫描结果
        report += "\n## 📈 数据文件扫描结果\n"
        
        if "error" in scan_result:
            report += f"- ❌ {scan_result['error']}\n"
        else:
            report += f"""- **扫描时间**: {scan_result['scan_time']}
- **扫描文件数**: {scan_result['sample_files']}
- **数据最早日期**: {scan_result.get('overall_min_date', 'N/A')}
- **数据最新日期**: {scan_result.get('overall_max_date', 'N/A')}

### 抽样文件详情:
"""
            for item in scan_result.get("date_ranges", [])[:5]:
                report += f"- `{item['file']}`: {item['min_date']} ~ {item['max_date']} ({item['rows']}行)\n"
                
        # 添加建议
        report += "\n## 💡 操作建议\n"
        
        if status.get("is_outdated", True):
            report += """- 🔄 数据已过时，建议立即更新：
  ```bash
  ./run_daily_update.sh -m update
  ```
"""
        else:
            report += "- ✅ 数据是最新的，无需更新\n"
            
        report += """
## 🔧 常用命令

```bash
# 检查数据状态
python3 data_update_tracker.py --check

# 查看更新历史
python3 data_update_tracker.py --history

# 手动更新数据
./run_daily_update.sh -m update
```
"""
        
        return report
        
    def get_update_history(self, limit: int = 10) -> List[Dict]:
        """获取更新历史"""
        if not self.update_log_file.exists():
            return []
            
        history = []
        with open(self.update_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines[-limit:]:
            try:
                history.append(json.loads(line.strip()))
            except:
                continue
                
        return history

def create_update_marker(data_dir: str = "full_securities_data"):
    """在数据目录创建更新标记文件"""
    marker_file = Path(data_dir) / "LAST_UPDATE.txt"
    
    content = f"""数据最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

这是一个自动生成的标记文件，用于追踪数据更新时间。
请不要手动修改或删除此文件。

查看详细状态请运行:
python3 data_update_tracker.py --check
"""
    
    with open(marker_file, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据更新状态追踪系统")
    parser.add_argument("--check", action="store_true", help="检查数据状态")
    parser.add_argument("--history", action="store_true", help="查看更新历史")
    parser.add_argument("--scan", action="store_true", help="扫描数据文件")
    parser.add_argument("--report", action="store_true", help="生成完整报告")
    parser.add_argument("--data-dir", default="full_securities_data", help="数据目录")
    
    args = parser.parse_args()
    
    tracker = DataUpdateTracker(args.data_dir)
    
    if args.check:
        is_fresh, status = tracker.check_data_freshness()
        print("\n📊 数据状态检查")
        print("-" * 50)
        
        if status.get("status") == "no_update_info":
            print("❌ 未找到更新信息")
        else:
            print(f"最后更新: {status['update_time']}")
            print(f"数据日期: {status['data_end_date']}")
            print(f"距今天数: {status['days_since_update']}天")
            print(f"数据状态: {'✅ 最新' if is_fresh else '⚠️ 需要更新'}")
            
    elif args.history:
        history = tracker.get_update_history()
        print("\n📜 更新历史")
        print("-" * 50)
        
        for record in history:
            print(f"{record['update_time']} - 数据至{record['data_end_date']} - 成功率{record['success_rate']}")
            
    elif args.scan:
        result = tracker.scan_data_dates()
        print("\n🔍 数据文件扫描结果")
        print("-" * 50)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    elif args.report:
        report = tracker.generate_status_report()
        
        # 保存报告
        report_file = Path("数据更新状态报告.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        print(f"\n报告已保存至: {report_file}")
        
    else:
        # 默认显示简要状态
        is_fresh, status = tracker.check_data_freshness()
        
        if is_fresh:
            print("✅ 数据是最新的")
        else:
            print("⚠️  数据需要更新")
            
        if status.get("data_end_date"):
            print(f"数据更新至: {status['data_end_date']}")

if __name__ == "__main__":
    main()