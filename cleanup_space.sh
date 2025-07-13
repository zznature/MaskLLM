#!/bin/bash

echo "=== 磁盘空间清理脚本 ==="
echo ""

echo "1. 清理临时文件..."
rm -rf /tmp/* 2>/dev/null || echo "Cannot clean /tmp"
rm -rf /var/tmp/* 2>/dev/null || echo "Cannot clean /var/tmp"

echo "2. 清理用户临时文件..."
find $HOME -name "*.tmp" -delete 2>/dev/null
find $HOME -name "*.log" -size +100M -delete 2>/dev/null

echo "3. 清理 Apptainer 缓存..."
apptainer cache clean 2>/dev/null || echo "Cannot clean apptainer cache"

echo "4. 清理 Python 缓存..."
find $HOME -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find $HOME -name "*.pyc" -delete 2>/dev/null

echo "5. 显示清理后的空间..."
df -h

echo ""
echo "清理完成！" 