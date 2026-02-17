#!/bin/bash
#
# StockTradebyZ Web应用启动脚本
#

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  StockTradebyZ Web应用启动脚本${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3，请先安装Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}Python版本:${NC} $PYTHON_VERSION"
echo ""

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}未找到虚拟环境，正在创建...${NC}"
    python3 -m venv venv

    echo -e "${YELLOW}安装依赖包...${NC}"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    echo -e "${GREEN}依赖安装完成！${NC}"
    echo ""
else
    echo -e "${GREEN}虚拟环境已存在${NC}"
    source venv/bin/activate
    echo ""
fi

# 检查数据库
if [ ! -f "../stock_data.db" ]; then
    echo -e "${YELLOW}警告: 未找到主数据库文件 stock_data.db${NC}"
    echo -e "${YELLOW}请确保在项目根目录存在该文件${NC}"
    echo ""
fi

# 启动Flask应用
echo -e "${GREEN}启动Flask应用...${NC}"
echo -e "${GREEN}访问地址: ${NC}http://127.0.0.1:5000"
echo ""
echo -e "${YELLOW}按Ctrl+C停止服务器${NC}"
echo ""

export FLASK_ENV=development
export FLASK_APP=app.py

python3 app.py
