---
title: Utilities
---

# Utilities API

## Auto Detection

!!! warning "`llm_rosetta.auto_detect` 已弃用"
    `llm_rosetta.auto_detect` 只是一层转发，自身没有任何逻辑：它把 0.13.0 的五个名称原样（同一对象）再导出，导入时发出警告，并将在 0.15.0 移除。请改为从 `llm_rosetta` 导入 `detect_provider`、`get_converter_for_provider`、`convert`、`convert_response` 和 `ProviderType`（参见[迁移到 0.14](../guide/migrating-to-0.14.md)）。

检测规则在格式目录中（`formats/detect.py`）；`detect_provider` 是入口。

::: llm_rosetta.converters.registry.detect_provider

## Types

::: llm_rosetta.contracts.routing.ProviderType

## Errors

::: llm_rosetta.pipeline.ConversionError
