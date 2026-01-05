# 安装

本页面将指导您如何安装 LLMIR。

## 系统要求

- Python 3.8 或更高版本
- pip 包管理器

## 使用 pip 安装

最简单的安装方式是使用 pip：

```bash
pip install llmir
```

## 从源码安装

如果您想要最新的开发版本，可以从 GitHub 仓库安装：

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir
pip install -e .
```

## 验证安装

安装完成后，您可以通过以下方式验证安装是否成功：

```python
import llmir
print(llmir.__version__)
```

## 可选依赖

LLMIR 的核心功能不需要额外的依赖，但某些高级功能可能需要安装额外的包：

```bash
# 用于开发和测试
pip install llmir[dev]

# 用于文档构建
pip install llmir[docs]

# 安装所有可选依赖
pip install llmir[all]
```

## 故障排除

如果您在安装过程中遇到问题，请查看以下常见解决方案：

### Python 版本问题

确保您使用的是 Python 3.8 或更高版本：

```bash
python --version
```

### 权限问题

如果遇到权限错误，可以使用用户安装模式：

```bash
pip install --user llmir
```

### 网络问题

如果网络连接有问题，可以使用国内镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple llmir
```

## 下一步

安装完成后，您可以：

- [学习基本用法](basic-usage.md)
- [查看示例代码](../examples/)
- [阅读 API 文档](../api/)