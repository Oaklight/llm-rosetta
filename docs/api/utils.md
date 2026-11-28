---
title: Utilities
---

# Utilities API

## Auto Detection

!!! warning "`llm_rosetta.auto_detect` is deprecated"
    `llm_rosetta.auto_detect` is a forwarding layer with no logic of its own: it re-exports the five names 0.13.0 held, as the same objects, and warns on import. It will be removed in 0.15.0. Import `detect_provider`, `get_converter_for_provider`, `convert`, `convert_response` and `ProviderType` from `llm_rosetta` instead (see [Migrating to 0.14](../guide/migrating-to-0.14.md)).

The detection rules live in the format catalog (`formats/detect.py`); `detect_provider` is the entry point.

::: llm_rosetta.converters.registry.detect_provider

## Types

::: llm_rosetta.contracts.routing.ProviderType

## Errors

::: llm_rosetta.pipeline.ConversionError
