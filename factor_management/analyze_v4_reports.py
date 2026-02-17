#!/usr/bin/env python3
"""
V4报告评分质量分析脚本
调用 analyze_quantitative_scoring_correlation.py 分析V4报告的评分质量
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

class V4ReportAnalyzer:
    """V4报告分析器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.project_root = Path(__file__).parent.parent
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("V4ReportAnalyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def check_reports_exist(self) -> dict:
        """检查V4报告是否存在"""
        v4_dir = self.project_root / "reports" / "daily_selection_v4"
        
        if not v4_dir.exists():
            return {'exists': False, 'message': f"V4报告目录不存在: {v4_dir}"}
        
        report_files = list(v4_dir.glob("V4选股分析报告_*.md"))
        
        if not report_files:
            return {'exists': False, 'message': f"V4报告目录中没有找到报告文件: {v4_dir}"}
        
        # 按文件名排序，获取日期范围
        report_files.sort()
        first_file = report_files[0].stem
        last_file = report_files[-1].stem
        
        return {
            'exists': True,
            'count': len(report_files),
            'directory': str(v4_dir),
            'first_report': first_file,
            'last_report': last_file,
            'message': f"找到 {len(report_files)} 个V4报告文件"
        }
    
    def run_analysis(self, start_date: str = None, end_date: str = None) -> bool:
        """运行V4报告分析"""
        
        # 检查报告是否存在
        report_check = self.check_reports_exist()
        if not report_check['exists']:
            self.logger.error(report_check['message'])
            return False
        
        self.logger.info(f"✅ {report_check['message']}")
        
        # 构建命令
        analysis_script = self.project_root / "analyze_quantitative_scoring_correlation.py"
        
        if not analysis_script.exists():
            self.logger.error(f"分析脚本不存在: {analysis_script}")
            return False
        
        cmd = [
            sys.executable,
            str(analysis_script),
            "--report-dir", "reports/daily_selection_v4",
            "--version", "v4"
        ]
        
        # 添加日期参数
        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])
        
        self.logger.info(f"🚀 开始运行V4报告分析...")
        self.logger.info(f"📝 命令: {' '.join(cmd)}")
        
        try:
            # 切换到项目根目录运行
            original_dir = os.getcwd()
            os.chdir(self.project_root)
            
            # 运行分析
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            
            # 切换回原目录
            os.chdir(original_dir)
            
            if result.returncode == 0:
                self.logger.info("✅ V4报告分析完成!")
                self.logger.info("📋 分析输出:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                
                return True
            else:
                self.logger.error(f"❌ 分析失败，返回码: {result.returncode}")
                self.logger.error("错误输出:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                
                return False
                
        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            self.logger.error("❌ 分析超时（30分钟）")
            return False
        except Exception as e:
            os.chdir(original_dir)
            self.logger.error(f"❌ 运行分析时出错: {e}")
            return False
    
    def show_analysis_results(self):
        """显示分析结果路径"""
        correlation_dir = self.project_root / "reports" / "correlation_analysis"
        
        if correlation_dir.exists():
            result_files = list(correlation_dir.glob("*v4*"))
            if result_files:
                print("\n📊 V4分析结果文件:")
                for file in sorted(result_files):
                    print(f"   📋 {file.name}")
                    
                print(f"\n📁 完整路径: {correlation_dir}")
            else:
                print(f"\n⚠️  在 {correlation_dir} 中未找到V4相关的分析结果")
        else:
            print(f"\n⚠️  分析结果目录不存在: {correlation_dir}")


def main():
    """主函数"""
    
    print("""
🔍 V4报告评分质量分析工具
==============================

本工具将分析V4选股报告中的评分与后续股价表现的相关性
    """)
    
    analyzer = V4ReportAnalyzer()
    
    try:
        # 运行分析
        success = analyzer.run_analysis()
        
        if success:
            print("=" * 60)
            print("🎉 V4报告分析完成！")
            print("=" * 60)
            
            # 显示结果文件
            analyzer.show_analysis_results()
            
            print("""
💡 分析内容包括:
   - V4评分与后续收益率相关性分析
   - 不同评分区间的表现统计
   - 风险调整收益率（夏普比率）分析
   - 相关性随时间的变化趋势
   - 可视化图表和统计报告

📈 结果解读:
   - 相关系数越高，说明V4评分对后续收益的预测能力越强
   - 夏普比率反映了风险调整后的收益质量
   - 时间序列分析显示评分系统的稳定性
            """)
            
            return 0
        else:
            print("❌ V4报告分析失败，请检查日志信息")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())