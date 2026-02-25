# 环境配置(CPU版本)
```bash
# CPU版本
# 将requirements.in编译为requirements.txt
uv pip compile requirements.in \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -o requirements.txt


# 安装虚拟环境
uv pip sync requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

```