# 开发指南

欢迎参与 LLMIR 的开发！本指南将帮助您了解如何为项目做出贡献。

## 开发环境设置

### 克隆仓库

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir
```

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

## 项目结构

```
llmir/
├── src/llmir/           # 主要源代码
│   ├── converters/      # 转换器实现
│   ├── types/          # 类型定义
│   └── utils/          # 工具函数
├── tests/              # 测试代码
├── docs/               # 文档
└── examples/           # 示例代码
```

## 贡献指南

### 报告问题

如果您发现了 bug 或有功能请求，请在 GitHub 上创建 issue。

### 提交代码

1. Fork 仓库
2. 创建功能分支
3. 编写代码和测试
4. 提交 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 添加类型注解
- 编写测试用例
- 更新文档

## 架构设计

了解 LLMIR 的核心架构和设计原则：

- [架构设计](architecture.md)
- [贡献指南](contributing.md)

## 下一步

- [查看贡献指南](contributing.md)
- [了解架构设计](architecture.md)
- [参与社区讨论](https://github.com/Oaklight/llmir/discussions)