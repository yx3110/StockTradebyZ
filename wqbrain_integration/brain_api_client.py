#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldQuant BRAIN API 客户端

功能:
1. 账号认证 + session 管理
2. 批量提交 alpha 表达式
3. 拉取回测结果 (Sharpe, IC, Turnover, Fitness)
4. 查询已提交 alpha 状态
5. Alpha 自动化挖掘流水线

依赖:
- requests (HTTP)
- 可选: pyworldquant (官方SDK, pip install pyworldquant)

BRAIN API 端点 (基于社区逆向):
- 认证: POST /api/v1/sessions
- 提交: POST /alphas
- 查询: GET /alphas/{id}
- 列表: GET /alphas?limit=N&offset=M
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

# BRAIN 平台域名
BRAIN_BASE_URL = 'https://api.worldquantbrain.com'


@dataclass
class AlphaResult:
    """Alpha 回测结果"""
    alpha_id: str = ''
    expression: str = ''
    status: str = 'pending'         # pending / running / done / error
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    ic: float = 0.0                 # Information Coefficient
    drawdown: float = 0.0
    is_pass: bool = False           # 是否通过 IS 检验
    raw_response: Dict = field(default_factory=dict)


class BrainAPIClient:
    """WorldQuant BRAIN API 客户端"""

    def __init__(self, credentials_path: str = None):
        """
        初始化客户端

        Args:
            credentials_path: 凭据文件路径, JSON 格式:
                {"email": "xxx@example.com", "password": "xxx"}
                默认: wqbrain_integration/credentials.json
        """
        self.base_url = BRAIN_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'StockTradebyZ-BrainClient/1.0',
        })
        self.authenticated = False
        self._credentials_path = credentials_path or str(
            Path(__file__).parent / 'credentials.json'
        )

        # 默认 alpha 设置
        # 注意: 中国区域需要在 BRAIN 平台开通后才能使用
        self.default_settings = {
            'instrumentType': 'EQUITY',
            'region': 'USA',
            'universe': 'TOP3000',
            'delay': 1,
            'decay': 5,
            'neutralization': 'SUBINDUSTRY',
            'truncation': 0.08,
            'nanHandling': 'ON',
            'pasteurization': 'ON',
            'unitHandling': 'VERIFY',
            'language': 'FASTEXPR',
            'visualization': False,
        }

        # 速率限制
        self._last_request_time = 0
        self._min_interval = 1.0  # 最小请求间隔(秒)

    # ----------------------------------------------------------
    # 认证
    # ----------------------------------------------------------

    def login(self, email: str = None, password: str = None) -> bool:
        """
        登录 BRAIN 平台

        Args:
            email: 邮箱 (优先级: 参数 > 凭据文件 > 环境变量)
            password: 密码
        """
        if not email or not password:
            email, password = self._load_credentials()

        if not email or not password:
            logger.error(
                "未找到 BRAIN 凭据. 请提供以下任一:\n"
                "  1. 参数: login(email='...', password='...')\n"
                "  2. 文件: wqbrain_integration/credentials.json\n"
                "  3. 环境变量: WQ_BRAIN_EMAIL, WQ_BRAIN_PASSWORD"
            )
            return False

        try:
            # BRAIN 使用 HTTP Basic Auth
            resp = self._request('POST', '/authentication',
                                 auth=(email, password))
            if resp.status_code in (200, 201):
                data = resp.json()
                # 检查是否需要 biometric 验证
                if 'inquiry' in data:
                    logger.warning(
                        "BRAIN 要求额外验证 (biometric/2FA). "
                        "请先在浏览器中登录 platform.worldquantbrain.com 完成验证."
                    )
                    return False
                self.authenticated = True
                logger.info(f"BRAIN 登录成功: {email}")
                return True
            else:
                logger.error(f"BRAIN 登录失败: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            logger.error(f"BRAIN 登录异常: {e}")
            return False

    def _load_credentials(self):
        """从文件或环境变量加载凭据"""
        # 优先环境变量
        email = os.environ.get('WQ_BRAIN_EMAIL')
        password = os.environ.get('WQ_BRAIN_PASSWORD')
        if email and password:
            return email, password

        # 凭据文件
        cred_path = Path(self._credentials_path)
        if cred_path.exists():
            with open(cred_path, 'r') as f:
                creds = json.load(f)
            return creds.get('email'), creds.get('password')

        return None, None

    # ----------------------------------------------------------
    # Alpha 提交
    # ----------------------------------------------------------

    def submit_alpha(self, expression: str,
                     settings: Dict = None,
                     name: str = None) -> Optional[str]:
        """
        提交一个 alpha 表达式到 BRAIN

        Args:
            expression: BRAIN FASTEXPR 表达式
            settings: alpha 设置 (合并到默认设置)
            name: alpha 名称 (可选)

        Returns:
            alpha_id 或 None (失败时)
        """
        if not self.authenticated:
            logger.error("未登录, 请先调用 login()")
            return None

        merged_settings = {**self.default_settings}
        if settings:
            merged_settings.update(settings)

        payload = {
            'type': 'REGULAR',
            'settings': merged_settings,
            'regular': expression,
        }

        try:
            # BRAIN 用 /simulations 提交回测
            # 成功返回 201 + Location header 指向 simulation URL
            # 需要轮询 Location URL 直到完成, 最终返回 alpha URL
            resp = self._request('POST', '/simulations', json=payload)
            if resp.status_code == 201:
                sim_url = resp.headers.get('Location', '')
                retry_after = float(resp.headers.get('Retry-After', '5'))
                if sim_url:
                    sim_id = sim_url.rstrip('/').split('/')[-1]
                    logger.info(f"Simulation 已提交: {sim_id} | {expression[:60]}...")
                    # 轮询等待 simulation 完成, 返回 alpha 短 ID
                    alpha_id = self._poll_simulation(sim_url, retry_after)
                    if alpha_id:
                        logger.info(f"Alpha 创建完成: {alpha_id}")
                        return alpha_id
                    return None
                logger.error("提交成功但未返回 Location")
                return None
            else:
                logger.error(f"提交失败: {resp.status_code} {resp.text[:300]}")
                return None
        except Exception as e:
            logger.error(f"提交异常: {e}")
            return None

    def _poll_simulation(self, sim_url: str, initial_retry: float = 5.0,
                         timeout: float = 300.0) -> Optional[str]:
        """轮询 simulation 直到完成, 返回 alpha ID"""
        start = time.time()
        wait = initial_retry

        while time.time() - start < timeout:
            time.sleep(wait)
            try:
                resp = self.session.get(sim_url)
                if resp.status_code != 200:
                    logger.error(f"Simulation 轮询失败: {resp.status_code}")
                    return None

                data = resp.json()

                # 检查进度
                progress = data.get('progress', None)
                if progress is not None and progress < 1.0:
                    logger.debug(f"Simulation 进度: {progress*100:.0f}%")
                    wait = 3.0
                    continue

                # 完成 — 状态为 COMPLETE 且有 alpha 短 ID
                status = data.get('status', '')
                alpha_id = data.get('alpha', '')
                if status == 'COMPLETE' and alpha_id:
                    return alpha_id

                # 其他完成情况
                if alpha_id:
                    return alpha_id

                logger.warning(f"Simulation 返回未预期格式: {list(data.keys())}")
                return None

            except Exception as e:
                logger.error(f"Simulation 轮询异常: {e}")
                return None

        logger.warning("Simulation 轮询超时")
        return None

    def batch_submit(self, alphas: List[Dict],
                     interval: float = 2.0) -> List[str]:
        """
        批量提交 alpha

        Args:
            alphas: [{'expression': '...', 'settings': {...}, 'name': '...'}, ...]
            interval: 提交间隔(秒), 避免触发频率限制

        Returns:
            alpha_id 列表
        """
        ids = []
        total = len(alphas)
        for i, alpha_cfg in enumerate(alphas, 1):
            alpha_id = self.submit_alpha(
                expression=alpha_cfg['expression'],
                settings=alpha_cfg.get('settings'),
                name=alpha_cfg.get('name'),
            )
            ids.append(alpha_id or '')
            logger.info(f"  [{i}/{total}] {'OK' if alpha_id else 'FAIL'}")
            if i < total:
                time.sleep(interval)
        return ids

    # ----------------------------------------------------------
    # 结果查询
    # ----------------------------------------------------------

    def get_alpha_result(self, alpha_id: str) -> AlphaResult:
        """查询 alpha 回测结果"""
        if not self.authenticated:
            return AlphaResult(alpha_id=alpha_id, status='error')

        try:
            resp = self._request('GET', f'/alphas/{alpha_id}')
            if resp.status_code == 200:
                data = resp.json()
                return self._parse_result(data)
            else:
                return AlphaResult(alpha_id=alpha_id, status='error')
        except Exception as e:
            logger.error(f"查询异常: {e}")
            return AlphaResult(alpha_id=alpha_id, status='error')

    def wait_for_results(self, alpha_ids: List[str],
                         poll_interval: float = 10.0,
                         timeout: float = 600.0) -> List[AlphaResult]:
        """
        等待一批 alpha 完成回测

        Args:
            alpha_ids: alpha ID 列表
            poll_interval: 轮询间隔(秒)
            timeout: 总超时(秒)
        """
        start_time = time.time()
        results = {aid: None for aid in alpha_ids if aid}

        while time.time() - start_time < timeout:
            all_done = True
            for aid in results:
                if results[aid] and results[aid].status == 'done':
                    continue
                result = self.get_alpha_result(aid)
                results[aid] = result
                if result.status not in ('done', 'error'):
                    all_done = False

            if all_done:
                break
            time.sleep(poll_interval)

        return [results.get(aid, AlphaResult(alpha_id=aid, status='timeout'))
                for aid in alpha_ids if aid]

    def list_alphas(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """列出已提交的 alpha"""
        if not self.authenticated:
            return []

        try:
            resp = self._request('GET', '/alphas',
                                 params={'limit': limit, 'offset': offset})
            if resp.status_code == 200:
                return resp.json().get('results', resp.json() if isinstance(resp.json(), list) else [])
            return []
        except Exception as e:
            logger.error(f"列表查询异常: {e}")
            return []

    # ----------------------------------------------------------
    # 结果分析
    # ----------------------------------------------------------

    def filter_passing_alphas(self, results: List[AlphaResult],
                              min_sharpe: float = 1.25,
                              max_turnover: float = 0.70,
                              min_fitness: float = 1.0) -> List[AlphaResult]:
        """筛选通过 IS 标准的 alpha"""
        passing = []
        for r in results:
            if (r.status == 'done'
                    and r.sharpe >= min_sharpe
                    and r.turnover <= max_turnover
                    and r.fitness >= min_fitness):
                r.is_pass = True
                passing.append(r)
        return passing

    def results_to_dataframe(self, results: List[AlphaResult]):
        """将结果转为 pandas DataFrame"""
        import pandas as pd
        rows = []
        for r in results:
            rows.append({
                'alpha_id': r.alpha_id,
                'expression': r.expression,
                'status': r.status,
                'sharpe': r.sharpe,
                'fitness': r.fitness,
                'turnover': r.turnover,
                'returns': r.returns,
                'ic': r.ic,
                'drawdown': r.drawdown,
                'is_pass': r.is_pass,
            })
        return pd.DataFrame(rows)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """带速率限制的请求"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f'{self.base_url}{path}'
        resp = self.session.request(method, url, **kwargs)
        self._last_request_time = time.time()
        return resp

    def _parse_result(self, data: Dict) -> AlphaResult:
        """解析 API 返回的 alpha 结果"""
        # BRAIN 实际格式: is/os/train/test/prod 分别是不同阶段的指标
        # IS (In-Sample) 是主要回测结果
        perf = data.get('is') or data.get('os') or {}

        # expression 可能是 dict {'code': '...', 'description': ...}
        expr = data.get('regular', '')
        if isinstance(expr, dict):
            expr = expr.get('code', '')

        # grade: SUPERIOR / GOOD / INFERIOR
        grade = data.get('grade', '')

        # 检查 IS checks 是否全部通过
        checks = perf.get('checks', [])
        all_pass = all(c.get('result') == 'PASS' for c in checks
                       if c.get('result') != 'PENDING')

        return AlphaResult(
            alpha_id=str(data.get('id', '')),
            expression=expr,
            status='done' if data.get('status') == 'UNSUBMITTED' or grade else
                   data.get('status', 'unknown'),
            sharpe=float(perf.get('sharpe', 0)),
            fitness=float(perf.get('fitness', 0)),
            turnover=float(perf.get('turnover', 0)),
            returns=float(perf.get('returns', 0)),
            ic=float(perf.get('ic', 0)),
            drawdown=float(perf.get('drawdown', 0)),
            is_pass=all_pass,
            raw_response=data,
        )


# ============================================================
# 便捷函数
# ============================================================

def quick_test_alpha(expression: str,
                     email: str = None, password: str = None,
                     region: str = 'CHINA',
                     wait: bool = True) -> Optional[AlphaResult]:
    """
    快速测试一个 alpha 表达式

    用法:
        result = quick_test_alpha('rank(volume / ts_mean(volume, 20))')
        print(f'Sharpe: {result.sharpe}, Fitness: {result.fitness}')
    """
    client = BrainAPIClient()
    if not client.login(email, password):
        return None

    alpha_id = client.submit_alpha(expression, settings={'region': region})
    if not alpha_id:
        return None

    if wait:
        results = client.wait_for_results([alpha_id])
        return results[0] if results else None
    return AlphaResult(alpha_id=alpha_id, status='pending')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='BRAIN API 客户端')
    sub = parser.add_subparsers(dest='cmd')

    # login
    login_p = sub.add_parser('login', help='测试登录')
    login_p.add_argument('--email', type=str)
    login_p.add_argument('--password', type=str)

    # submit
    submit_p = sub.add_parser('submit', help='提交 alpha')
    submit_p.add_argument('expression', type=str, help='BRAIN 表达式')
    submit_p.add_argument('--region', default='CHINA')
    submit_p.add_argument('--wait', action='store_true', help='等待回测完成')

    # submit-file
    file_p = sub.add_parser('submit-file', help='从 JSON 文件批量提交')
    file_p.add_argument('file', type=str, help='JSON 文件路径')
    file_p.add_argument('--interval', type=float, default=2.0)

    # list
    list_p = sub.add_parser('list', help='列出已提交 alpha')
    list_p.add_argument('--limit', type=int, default=20)

    # query
    query_p = sub.add_parser('query', help='查询 alpha 结果')
    query_p.add_argument('alpha_id', type=str)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    client = BrainAPIClient()

    if args.cmd == 'login':
        ok = client.login(args.email, args.password)
        print('登录成功' if ok else '登录失败')

    elif args.cmd == 'submit':
        if not client.login():
            print('请先配置凭据')
        else:
            aid = client.submit_alpha(args.expression, settings={'region': args.region})
            if aid:
                print(f'Alpha ID: {aid}')
                if args.wait:
                    results = client.wait_for_results([aid])
                    r = results[0]
                    print(f'Sharpe: {r.sharpe:.3f} | Fitness: {r.fitness:.3f} | '
                          f'Turnover: {r.turnover:.3f} | IC: {r.ic:.4f}')

    elif args.cmd == 'submit-file':
        if not client.login():
            print('请先配置凭据')
        else:
            with open(args.file, 'r') as f:
                alphas = json.load(f)
            ids = client.batch_submit(alphas, interval=args.interval)
            print(f'提交 {len(ids)} 个 alpha, 成功 {sum(1 for i in ids if i)}')

    elif args.cmd == 'list':
        if not client.login():
            print('请先配置凭据')
        else:
            alphas = client.list_alphas(limit=args.limit)
            for a in alphas:
                print(f"  {a.get('id', 'N/A'):>10s} | {a.get('regular', '')[:60]}")

    elif args.cmd == 'query':
        if not client.login():
            print('请先配置凭据')
        else:
            r = client.get_alpha_result(args.alpha_id)
            print(f'Status: {r.status}')
            print(f'Sharpe: {r.sharpe:.3f} | Fitness: {r.fitness:.3f} | '
                  f'Turnover: {r.turnover:.3f} | IC: {r.ic:.4f}')

    else:
        parser.print_help()
