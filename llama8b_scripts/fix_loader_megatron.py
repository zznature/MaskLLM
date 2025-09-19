#!/usr/bin/env python3

# 临时修复 loader_megatron.py 中的解包问题
# 这个脚本会创建一个修复版本的loader文件

import shutil
import os

original_file = "/data/home/zdhs0054/zzhou/MaskLLM/tools/checkpoint/loader_megatron.py"
backup_file = "/data/home/zdhs0054/zzhou/MaskLLM/tools/checkpoint/loader_megatron.py.backup"

# 备份原文件
if not os.path.exists(backup_file):
    shutil.copy2(original_file, backup_file)
    print(f"✅ 已备份原文件: {backup_file}")

# 读取原文件
with open(original_file, 'r') as f:
    content = f.read()

# 修复第60行的解包问题
old_line = "    margs, checkpoint_args = load_args_from_checkpoint(margs)"
new_line = """    result = load_args_from_checkpoint(margs)
    if isinstance(result, tuple):
        margs, checkpoint_args = result
    else:
        margs = result
        checkpoint_args = None"""

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # 写回文件
    with open(original_file, 'w') as f:
        f.write(content)
    
    print(f"✅ 已修复 loader_megatron.py 第60行的解包问题")
    print("修复内容：处理 load_args_from_checkpoint 返回值不一致的情况")
else:
    print("❌ 未找到需要修复的行，文件可能已经被修改")

print("\n修复完成，现在可以重试TP转换：")
print("bash run_maskllm_native.sh llama8b_scripts/convert_llama8b_tp8_to_tp1.sh")
