"""
Tests for BaseConverter
"""

import pytest

from llmir.converters.base import BaseConverter


class MockConverter(BaseConverter):
    """用于测试的 Mock Converter"""

    def to_provider(self, ir_input, tools=None, tool_choice=None):
        return {}, []

    def from_provider(self, provider_data):
        return []


class TestBaseConverter:
    """测试 BaseConverter 的验证功能"""

    def setup_method(self):
        """设置测试"""
        self.converter = MockConverter()

    def test_validate_ir_input_valid_messages(self):
        """测试有效的消息列表"""
        ir_input = [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {"role": "user", "content": [{"type": "text", "text": "User message"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant message"}],
            },
        ]
        errors = self.converter.validate_ir_input(ir_input)
        assert errors == []

    def test_validate_ir_input_developer_role(self):
        """测试 developer 角色"""
        ir_input = [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Developer message"}],
            }
        ]
        errors = self.converter.validate_ir_input(ir_input)
        assert errors == []

    def test_validate_ir_input_not_list(self):
        """测试输入不是列表"""
        ir_input = {"role": "user", "content": []}
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "must be a list" in errors[0]

    def test_validate_ir_input_item_not_dict(self):
        """测试列表项不是字典"""
        ir_input = ["not a dict", {"role": "user", "content": []}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]

    def test_validate_ir_input_invalid_role(self):
        """测试无效的角色"""
        ir_input = [{"role": "invalid_role", "content": []}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "invalid role" in errors[0]

    def test_validate_ir_input_missing_content(self):
        """测试缺少 content 字段"""
        ir_input = [{"role": "user"}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "missing required 'content' field" in errors[0]

    def test_validate_ir_input_content_not_list(self):
        """测试 content 不是列表"""
        ir_input = [{"role": "user", "content": "not a list"}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "'content' must be a list" in errors[0]

    def test_validate_ir_input_valid_extension_items(self):
        """测试有效的扩展项"""
        ir_input = [
            {"type": "system_event", "event_type": "start"},
            {"type": "batch_marker", "batch_id": "123"},
            {"type": "session_control", "action": "reset"},
            {"type": "tool_chain_node", "node_id": "node1"},
        ]
        errors = self.converter.validate_ir_input(ir_input)
        assert errors == []

    def test_validate_ir_input_invalid_extension_type(self):
        """测试无效的扩展项类型"""
        ir_input = [{"type": "invalid_extension"}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "invalid type" in errors[0]

    def test_validate_ir_input_item_without_role_or_type(self):
        """测试既没有 role 也没有 type 的项"""
        ir_input = [{"some_field": "value"}]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 1
        assert "must have either 'role'" in errors[0]

    def test_validate_ir_input_multiple_errors(self):
        """测试多个错误"""
        ir_input = [
            {"role": "invalid_role", "content": []},
            {"role": "user"},
            {"type": "invalid_type"},
        ]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 3

    def test_validate_ir_input_mixed_valid_and_invalid(self):
        """测试混合有效和无效的项"""
        ir_input = [
            {"role": "user", "content": [{"type": "text", "text": "Valid"}]},
            {"role": "invalid_role", "content": []},
            {"type": "system_event", "event_type": "test"},
            {"some_field": "no role or type"},
        ]
        errors = self.converter.validate_ir_input(ir_input)
        assert len(errors) == 2
