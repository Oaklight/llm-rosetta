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

    # ==================== 分层抽象方法实现 Layered abstract method implementations ====================

    def _ir_message_to_p(self, message, ir_input):
        return {}

    def _ir_content_part_to_p(self, content_part, ir_input):
        return {}

    def _p_message_to_ir(self, provider_message):
        return {}

    def _p_content_part_to_ir(self, provider_part):
        return []

    # ==================== 类型特定转换方法实现 Type-specific conversion method implementations ====================

    def _ir_text_to_p(self, text_part):
        return {}

    def _p_text_to_ir(self, provider_text):
        return {"type": "text", "text": ""}

    def _ir_image_to_p(self, image_part):
        return {}

    def _p_image_to_ir(self, provider_image):
        return {"type": "image", "image_url": ""}

    def _ir_file_to_p(self, file_part):
        return {}

    def _p_file_to_ir(self, provider_file):
        return {"type": "file", "file_url": ""}

    def _ir_tool_call_to_p(self, tool_call_part):
        return {}

    def _p_tool_call_to_ir(self, provider_tool_call):
        return {
            "type": "tool_call",
            "tool_call_id": "",
            "tool_name": "",
            "tool_input": {},
        }

    def _ir_tool_result_to_p(self, tool_result_part):
        return {}

    def _p_tool_result_to_ir(self, provider_tool_result):
        return {"type": "tool_result", "tool_call_id": "", "result": ""}

    def _ir_tool_to_p(self, tool):
        return {}

    def _p_tool_to_ir(self, provider_tool):
        return {"type": "function", "name": "", "description": "", "parameters": {}}

    def _ir_tool_choice_to_p(self, tool_choice):
        return {}

    def _p_tool_choice_to_ir(self, provider_tool_choice):
        return {"mode": "auto"}


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
        assert "must be an iterable" in errors[0]

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
