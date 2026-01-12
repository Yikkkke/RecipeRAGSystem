# 开发环境说明（ENV.md）

本仓库支持 **本地电脑** 与 **GitHub Codespaces** 两种开发方式，统一通过 GitHub 仓库同步代码。

> 设计原则：
> - **环境可复现**（不依赖某一台机器）
> - **避免 CUDA / 大体积依赖**（对 Codespaces 友好）
> - **子项目独立虚拟环境**（避免相互污染）

---
## 0、常用命令

### 开发前，拉取最新代码
```bash
git pull
```

### 开发完成，一定要push到GitHub仓库
```bash
git status
git add .  或者 git add filename.py
git commit -m "feat: 本地完善 RAG 示例代码"
git push

```
提示：
feat: 新功能
fix: 修 bug
chore: 杂项（配置、文档）

### 版本号git管理
```bash
# 以提交到main为例，提交到分支时将main改成对应分支名
## 开发前拉取新代码
git checkout main
git pull
## 开发完成后push代码
git add .
git commit -m "feat: RecipeRAGSystem v0.1 production baseline"

git tag -a v0.1.0 -m "v0.1.0: baseline RAG system with vector index"
git push origin main
git push origin v0.1.0
```


### windows环境变量配置（不被暴露到GitHub仓库！！
在项目的根目录下添加.env文件，用来本地防止api_key等局部环境变量
```.env
MOONSHOT_API_KEY = 26138716498389402
```
在项目的根目录下的.gitignore文件中，添加一行
```
.env
```

### 其他常用命令
重新编译生成requirements.txt : uv pip compile requirements.in -o requirements.txt --no-cache
环境配置命令codespaces：uv pip install -r requirements.txt --no-cache
查看磁盘空间：du -h -d 1 ~ | sort -h
清空缓存释放空间：rm -rf ~/.cache/uv         rm -rf ~/.cache/pip

运行Python文件：uv run python main.py


## 一、推荐开发模式（结论先行）

- ✅ **本地电脑：主力开发**（性能最好、磁盘充足）
- ✅ **Codespaces：临时 / 公司电脑 / 小改动**
- ✅ **所有状态通过 Git 同步**（不提交虚拟环境）

---

## 二、仓库结构与环境策略

```text
all-in-rag/
├── code/
│   ├── C1/
│   ├── C2/
│   ├── C3/
│   └── C8-reproduction/   # ⭐ 独立虚拟环境
│       ├── requirements.in
│       ├── requirements.txt
│       └── rag_modules/
└── data/                  # 数据目录（不提交）
```

### 约定

- **C8-reproduction 单独维护虚拟环境**
- 其他子目录默认不要求可直接运行
- `.venv/` 永不提交到 Git

---

## 三、本地电脑开发（推荐）

### 1️⃣ 克隆仓库

```bash
git clone https://github.com/<your-org>/all-in-rag.git
cd all-in-rag/code/C8-reproduction
```

### 2️⃣ 创建虚拟环境（Python 3.12）

```bash
uv venv .venv --python 3.12
```

### 3️⃣ 安装依赖（CPU-only）

```bash
uv pip install -r requirements.txt
```

### 4️⃣ 验证环境

```bash
uv run python - << 'EOF'
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

期望输出：

```text
CUDA available: False
```

---

## 四、GitHub Codespaces 开发（磁盘受限）

### ⚠️ 注意事项

- Codespaces **磁盘空间有限**
- 禁止安装 CUDA / GPU 相关依赖
- 必须关闭安装缓存

### 1️⃣ 进入 Codespaces 后

```bash
cd code/C8-reproduction
rm -rf .venv
```

### 2️⃣ 创建虚拟环境

```bash
uv venv .venv --python 3.12
```

### 3️⃣ 安装依赖（必须加 --no-cache）

```bash
uv pip install -r requirements.txt --no-cache
```

### 4️⃣ 安装过程中警告

❌ 如果看到以下包，**立刻中止安装**：

```text
nvidia-cudnn-cu12
nvidia-cublas-cu12
nvidia-cusolver-cu12
```

说明 CUDA 被错误解析，需要重新生成 `requirements.txt`。

---

## 五、依赖管理规范（重要）

### 文件说明

- `requirements.in`：
  - 意图级依赖
  - 手工维护

- `requirements.txt`：
  - 由 uv 自动生成
  - **唯一安装入口**

### 更新依赖流程

```bash
uv pip compile requirements.in \
  -o requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
```

> ⚠️ 必须指定 CPU-only PyTorch index

---

## 六、Git 使用约定

### 不提交的内容

```gitignore
.venv/
.env
__pycache__/
data/
*.pdf
*.png
volumes/
```

### 推荐 commit 语义

- `feat(c8): ...` 新功能
- `fix(c8): ...` 修复问题（如依赖 / 环境）
- `chore(c8): ...` 构建、整理、文档

---

## 七、常见问题（FAQ）

### Q1：为什么不用 GPU？

- Codespaces 无 GPU
- CUDA 依赖体积巨大（>1GB）
- 本项目以 **RAG 结构 / 工程实践** 为主

---

### Q2：换电脑后怎么恢复环境？

```bash
git clone
cd code/C8-reproduction
uv venv .venv --python 3.12
uv pip install -r requirements.txt
```

---

## 八、一句话总结

> **环境不是状态，Git 才是**  
> **依赖要工程化，CUDA 要警惕**  
> **本地为主，Codespaces 为辅**

