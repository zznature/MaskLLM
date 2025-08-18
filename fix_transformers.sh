#!/bin/bash

set -euo pipefail

echo "=== Transformers 与相关依赖安装脚本（使用镜像，低空间友好） ==="
echo ""

# 镜像与环境设置
MIRROR_PYPI_URL=${MIRROR_PYPI_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_INDEX_URL="${MIRROR_PYPI_URL}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-180}
export PATH="$HOME/.local/bin:$PATH"

# 将临时目录与缓存目录放到可用空间较大的位置
TMPDIR_DEFAULT="/data/home/zdhs0054/zzhou/MaskLLM/.tmp/pip"
export TMPDIR=${TMPDIR:-$TMPDIR_DEFAULT}
mkdir -p "$TMPDIR" || true

echo "0. 镜像设置："
echo "  - PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "  - HF_ENDPOINT=${HF_ENDPOINT}"
echo "  - TMPDIR=${TMPDIR}"
echo "  - PATH 添加: $HOME/.local/bin"
echo ""

echo "1. 强制卸载transformers（如果存在）："
pip3 uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"
pip uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"
echo ""

echo "2. 清理可能的残留文件："
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无残留目录"
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无残留文件"
echo ""

echo "3. 清理用户目录下的残留："
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无用户目录残留"
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无用户目录残留"
echo ""

echo "4. 可选：升级pip（使用清华镜像，用户目录）"
python3 -m pip install -i "${MIRROR_PYPI_URL}" --user --upgrade pip --no-cache-dir || true
echo ""

echo "5. 安装指定包（使用清华镜像，安装到用户目录，优先二进制，不缓存）："
echo "   - transformers==4.40 accelerate datasets sentencepiece wandb tqdm ninja tensorboardx==2.6 pulp einops nltk pydantic==1.10.8"
python3 -m pip install -i "${MIRROR_PYPI_URL}" --user --no-cache-dir --prefer-binary \
  "transformers==4.40" accelerate datasets sentencepiece wandb tqdm ninja \
  "tensorboardx==2.6" pulp einops nltk "pydantic==1.10.8"
echo ""

echo "6. 安装 timm（不拉取巨型依赖，避免空间不足）："
echo "   - 首先尝试无依赖安装 timm（若缺少 torch，后续可按需单独安装）"
python3 -m pip install -i "${MIRROR_PYPI_URL}" --user --no-cache-dir --prefer-binary --no-deps timm || echo "timm 无依赖安装失败，可稍后单独安装"
echo ""

echo "7. 验证安装与可用性："
python3 - <<'PY'
import importlib, sys
mods = [
    ("transformers", "__version__"),
    ("accelerate", "__version__"),
    ("datasets", "__version__"),
    ("sentencepiece", None),
    ("wandb", "__version__"),
    ("tqdm", "__version__"),
    ("ninja", None),
    ("tensorboardX", "__version__"),
    ("pulp", None),
    ("timm", "__version__"),
    ("einops", "__version__"),
    ("nltk", "__version__"),
    ("pydantic", "__version__"),
]
ok = True
for name, ver_attr in mods:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, ver_attr) if ver_attr else "OK"
        print(f"[OK] {name}: {ver}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")
sys.exit(0 if ok else 1)
PY
echo ""

echo "8. PATH 提示：如需在容器任意位置调用安装的命令，请将如下加入 ~/.bashrc："
echo '   export PATH="$HOME/.local/bin:$PATH"'
echo ""

echo "=== 安装完成 ==="