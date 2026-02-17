#!/usr/bin/env python3
"""
雪球API配置文件
用于设置雪球cookie和其他认证信息
"""

# 雪球cookie配置
# 如果你有雪球的登录cookie，可以在这里设置
# 格式: "cookie_name1=value1; cookie_name2=value2; ..."
XUEQIU_COOKIE = "cookiesu=111746700523129; device_id=7718cf732d4ead0ff874930ef75bcba4; s=c012fi9trd; bid=052ea58839559647471af1a1af781f4b_maf8fjsx; xq_a_token=84e219bf8570a248762ec7d625cc958ce34564e4; xqat=84e219bf8570a248762ec7d625cc958ce34564e4; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjE3MzQyMjU4MTUsImlzcyI6InVjIiwiZXhwIjoxNzU2NDMyNTkyLCJjdG0iOjE3NTQwNTQ5NDUxNzMsImNpZCI6ImQ5ZDBuNEFadXAifQ.bOe84h-5dY1fz1jP7A5uvwG2XGUt-WBExeymMYtMqDXawBTzmFo23PgFsKunJF4enTnIRPxOHlgIRXcMzOLTjqdNEl8ZYdDOzRKnEKNGjVJTYxJiSM1eRVPlGPdaA1_ui_a74GNfrL2mAjv7DJ6HXtSSFB55E1-iPTVJhLCa5WKPrdvpQN-ysLHgs8zNMnlfbLuj5Z7KIIwVU_AHaKy02gHZc5A5XzIS9tfKsWLPCrTwCiMy3S2fLqHavuXgQLmtiZjlXfP6jAU5rG-uCJ8Mpb8nyunYp2G30Lr4QlM7TZobo02KxY_7AI1Tnlhc9b-UaCEGmI8sdnDPEpPJ242PeQ; xq_r_token=caf617e3ff3259a06ae580eed716175ee761c20b; xq_is_login=1; u=1734225815; acw_tc=0a099d7617543977179198065e35f056948db90da61adc269210eb964f874a; ssxmod_itna=eqIhGKiKAKYvVD4knD9exIxUE4YL2DBP01DpxYK0C1DLxnb4Gde2DBnPtA6CinCQ7DD7fWOiAPdnmqDsqFe4GzDiuPGhDBWYHleIL+Khdzt0O4==tRA+iv5tnthi2tojYxu8cCtKVS9qlFCd0F1yP4xhDiTrD07DmeDUxD1pDDkgBPDxhPD5xDTDWeDGDD384DCA8OoD0RAC43E3FRmr7gADYaor9roDY50DAqjfRSA3tDRrDDzX4iarI4fpnYDEhhcpn2AD7P0E7=DXPoDE0Wv+dtDv1jm=KwEEU8op5b450e/hBa4cGW1Ki=/hiYiPt0ZSuNfDxzbt804QAKmTqR2ZjAHDDWYf4Y+shDY1Q1ju1r4zhBsQisrWsQE4Mx3e7+jhicD=h04g75ZAxc7DGo+52xFBDTlizEh4D; ssxmod_itna2=eqIhGKiKAKYvVD4knD9exIxUE4YL2DBP01DpxYK0C1DLxnb4Gde2DBnPtA6CinCQ7DD7fWOiAPdne4DWhfaOEb4DFoAKficDGXKdjFayG658Ym8+zWNe1MGeLEUG09P0VFEIkjvNYmQavriPk19DWrDHUAFaalvKGipdeWuH/h0C5jcm7=iPYbvdqo5AP8aEqCD8YfO47apxUaa=p4HxjUgkAxkA9ovfYjE4L17j+9GaXKff0PjtbtD6oWgFeCB+YI0FaWpdnKjwb9j9TwYU7ZOjpLD08Wwzf3308VEdrafApNOPffhKZjou7t4jG=pDBYY6=EDhAKW58ajqe+dEx3pGCCIeYeyOPU6KhmYdSIqErtSE0WGfp2=4rNzG8C2iYeObTdmw1nwKWqN6qLruOz2wYexD"

# 使用说明:
# 1. 打开浏览器，登录雪球网站 (https://xueqiu.com)
# 2. 按F12打开开发者工具
# 3. 在Network标签页中，刷新页面
# 4. 找到对xueqiu.com的请求，复制Request Headers中的Cookie值
# 5. 将Cookie值设置到上面的XUEQIU_COOKIE变量中
# 
# 示例:
# XUEQIU_COOKIE = "acw_tc=xxxx; xq_a_token=xxxx; xqat=xxxx; xq_r_token=xxxx; xq_id_token=xxxx"

def get_xueqiu_cookie():
    """获取雪球cookie配置"""
    return XUEQIU_COOKIE

def set_xueqiu_cookie(cookie_string: str):
    """动态设置雪球cookie"""
    global XUEQIU_COOKIE
    XUEQIU_COOKIE = cookie_string
    print(f"雪球cookie已更新")

# 验证cookie是否有效的简单检查
def validate_cookie_format(cookie_string: str) -> bool:
    """验证cookie格式是否正确"""
    if not cookie_string:
        return False
    
    # 基本格式检查
    if '=' not in cookie_string:
        return False
    
    # 检查是否包含关键的雪球认证字段
    # 根据研究，xq_a_token是最重要的认证token
    required_fields = ['xq_a_token']
    has_required = any(field in cookie_string for field in required_fields)
    
    if not has_required:
        print("⚠️ 警告: Cookie中缺少关键的xq_a_token字段")
        print("   这可能导致400016认证错误")
        print("   请确保从登录后的浏览器中复制完整的Cookie")
    
    return has_required

def extract_xq_a_token(cookie_string: str) -> str:
    """从cookie字符串中提取xq_a_token"""
    if not cookie_string:
        return ""
    
    # 查找xq_a_token
    import re
    match = re.search(r'xq_a_token=([^;]+)', cookie_string)
    if match:
        return match.group(1)
    return ""

if __name__ == "__main__":
    # 测试配置
    cookie = get_xueqiu_cookie()
    if cookie:
        print(f"雪球cookie已配置，长度: {len(cookie)}")
        print(f"格式验证: {'通过' if validate_cookie_format(cookie) else '失败'}")
    else:
        print("雪球cookie未配置")
        print("\n如需使用雪球数据，请按以下步骤配置cookie:")
        print("1. 浏览器登录 https://xueqiu.com")
        print("2. F12 -> Network -> 刷新页面")
        print("3. 复制Request Headers中的Cookie")
        print("4. 在本文件中设置XUEQIU_COOKIE变量")