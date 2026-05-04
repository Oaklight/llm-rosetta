# /// zerodep
# version = "0.4.3"
# deps = []
# tier = "medium"
# category = "validation"
# note = "Install/update via `zerodep add validate`"
# ///

"""Zero-dependency runtime validator for TypedDict and dataclass types.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Validate arbitrary data against stdlib type annotations (TypedDict,
dataclass, Annotated constraints) and generate JSON Schema from the
same type definitions.

Basic usage::

    from validate import validate, json_schema, ValidationError

    class User(TypedDict):
        name: str
        age: int

    validate({"name": "Alice", "age": 30}, User)   # ok
    validate({"name": "Alice", "age": "x"}, User)   # raises ValidationError

    schema = json_schema(User)  # {"type": "object", "properties": ...}

Annotated constraints::

    from validate import Gt, MinLen
    from typing import Annotated

    class Item(TypedDict):
        name: Annotated[str, MinLen(1)]
        price: Annotated[float, Gt(0)]

Field validators (transform + validate)::

    from validate import FieldValidator

    def strip_lower(v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("must not be empty")
        return v

    class User(TypedDict):
        name: Annotated[str, FieldValidator(strip_lower)]

Model validators (cross-field validation)::

    from validate import model_validator

    class RegisterForm(TypedDict):
        password: str
        confirm: str

    @model_validator(RegisterForm)
    def passwords_match(data: dict) -> dict:
        if data["password"] != data["confirm"]:
            raise ValueError("passwords do not match")
        return data
"""

from __future__ import annotations

import dataclasses
import functools
import re
import typing
from collections.abc import Callable
from typing import Any, Union, get_type_hints

__all__ = [
    # Constraint annotations
    "Gt",
    "Ge",
    "Lt",
    "Le",
    "MinLen",
    "MaxLen",
    "Match",
    "Predicate",
    "FieldValidator",
    # Error types
    "ErrorDetail",
    "ValidationError",
    # Public API
    "validate",
    "json_schema",
    "model_validator",
]

# ── Constraint Annotations ──


@dataclasses.dataclass(frozen=True, slots=True)
class Gt:
    """Value must be strictly greater than *val*."""

    val: float

    def check(self, value: Any) -> bool:
        return value > self.val

    def schema_kw(self) -> dict[str, Any]:
        return {"exclusiveMinimum": self.val}

    def __str__(self) -> str:
        return f"> {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class Ge:
    """Value must be greater than or equal to *val*."""

    val: float

    def check(self, value: Any) -> bool:
        return value >= self.val

    def schema_kw(self) -> dict[str, Any]:
        return {"minimum": self.val}

    def __str__(self) -> str:
        return f">= {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class Lt:
    """Value must be strictly less than *val*."""

    val: float

    def check(self, value: Any) -> bool:
        return value < self.val

    def schema_kw(self) -> dict[str, Any]:
        return {"exclusiveMaximum": self.val}

    def __str__(self) -> str:
        return f"< {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class Le:
    """Value must be less than or equal to *val*."""

    val: float

    def check(self, value: Any) -> bool:
        return value <= self.val

    def schema_kw(self) -> dict[str, Any]:
        return {"maximum": self.val}

    def __str__(self) -> str:
        return f"<= {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class MinLen:
    """Length must be at least *val*."""

    val: int

    def check(self, value: Any) -> bool:
        return len(value) >= self.val

    def schema_kw(self) -> dict[str, Any]:
        if isinstance(self.val, int):
            return {"minLength": self.val}
        return {"minLength": self.val}

    def schema_kw_array(self) -> dict[str, Any]:
        return {"minItems": self.val}

    def __str__(self) -> str:
        return f"len >= {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class MaxLen:
    """Length must be at most *val*."""

    val: int

    def check(self, value: Any) -> bool:
        return len(value) <= self.val

    def schema_kw(self) -> dict[str, Any]:
        return {"maxLength": self.val}

    def schema_kw_array(self) -> dict[str, Any]:
        return {"maxItems": self.val}

    def __str__(self) -> str:
        return f"len <= {self.val}"


@dataclasses.dataclass(frozen=True, slots=True)
class Match:
    """Value must match *pattern* (via ``re.fullmatch``)."""

    pattern: str

    def check(self, value: Any) -> bool:
        return re.fullmatch(self.pattern, value) is not None

    def schema_kw(self) -> dict[str, Any]:
        return {"pattern": self.pattern}

    def __str__(self) -> str:
        return f"match({self.pattern!r})"


@dataclasses.dataclass(frozen=True, slots=True)
class Predicate:
    """Value must satisfy a custom predicate function.

    Args:
        fn: A callable ``(value) -> bool``.
        description: Human-readable description for error messages.
    """

    fn: Callable[[Any], bool]
    description: str = "custom predicate"

    def check(self, value: Any) -> bool:
        return self.fn(value)

    def schema_kw(self) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return self.description


@dataclasses.dataclass(frozen=True, slots=True)
class FieldValidator:
    """Custom validator that can transform the field value.

    Unlike ``Predicate`` (which returns bool), the function receives the
    validated value and returns a (possibly transformed) value.  Raise
    ``ValueError`` or ``AssertionError`` to signal failure.

    Args:
        fn: A callable ``(value) -> value``.  Raise on failure.
        description: Human-readable description for error messages.
    """

    fn: Callable[[Any], Any]
    description: str = "custom validator"

    def validate(self, value: Any) -> Any:
        return self.fn(value)

    def schema_kw(self) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return self.description


# Constraint base types for isinstance checks
_CONSTRAINT_TYPES = (Gt, Ge, Lt, Le, MinLen, MaxLen, Match, Predicate, FieldValidator)


# ── Model Validator Registry ──

_MODEL_VALIDATORS: dict[type, list[Callable]] = {}


def model_validator(tp: type) -> Callable[[Callable], Callable]:
    """Register a model-level validator for a TypedDict or dataclass type.

    The validator receives the full data dict after all field validation
    passes.  It should return the (possibly modified) dict, or raise
    ``ValueError`` / ``AssertionError`` on failure.

    Args:
        tp: The TypedDict or dataclass type to attach the validator to.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Example::

        class RegisterForm(TypedDict):
            password: str
            confirm: str

        @model_validator(RegisterForm)
        def passwords_match(data: dict) -> dict:
            if data["password"] != data["confirm"]:
                raise ValueError("passwords do not match")
            return data
    """

    def decorator(fn: Callable) -> Callable:
        _MODEL_VALIDATORS.setdefault(tp, []).append(fn)
        return fn

    return decorator


# ── Error Types ──


@dataclasses.dataclass
class ErrorDetail:
    """A single validation error.

    Attributes:
        path: Dotted/bracketed path to the failing field (e.g. ``"items[2].name"``).
        expected: Expected type or constraint description.
        actual: Actual type or value description.
        message: Human-readable error message.
    """

    path: str
    expected: str
    actual: str
    message: str


class ValidationError(Exception):
    """Raised when validation fails.

    Attributes:
        errors: List of all validation errors found.
    """

    def __init__(self, errors: list[ErrorDetail]) -> None:
        self.errors = errors
        msgs = "; ".join(e.message for e in errors[:5])
        if len(errors) > 5:
            msgs += f" ... and {len(errors) - 5} more"
        super().__init__(f"{len(errors)} validation error(s): {msgs}")


# ── Internal Helpers ──


@functools.cache
def _unwrap_annotated(tp: Any) -> tuple[Any, tuple[Any, ...]]:
    """Extract the base type and constraint metadata from an Annotated type.

    Results are cached because Annotated type structures are static at
    runtime.

    Returns:
        ``(base_type, (constraint1, constraint2, ...))`` if Annotated,
        otherwise ``(tp, ())``.
    """
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        args = typing.get_args(tp)
        base = args[0]
        constraints = tuple(a for a in args[1:] if isinstance(a, _CONSTRAINT_TYPES))
        return base, constraints
    return tp, ()


@functools.cache
def _is_typeddict(tp: Any) -> bool:
    """Check if *tp* is a TypedDict class.

    Uses ``__required_keys__`` attribute detection to support TypedDicts
    created with both ``typing.TypedDict`` and ``typing_extensions.TypedDict``.

    Results are cached because type identity is stable at runtime.
    """
    return isinstance(tp, type) and hasattr(tp, "__required_keys__")


@functools.cache
def _is_dataclass_type(tp: Any) -> bool:
    """Check if *tp* is a dataclass class (not an instance).

    Results are cached because type identity is stable at runtime.
    """
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _strip_required(tp: Any) -> Any:
    """Strip ``Required`` / ``NotRequired`` wrappers, returning the inner type."""
    import sys

    origin = typing.get_origin(tp)
    if sys.version_info >= (3, 11):
        from typing import NotRequired, Required
    else:
        from typing_extensions import NotRequired, Required
    if origin is Required or origin is NotRequired:
        args = typing.get_args(tp)
        return args[0] if args else tp
    return tp


@functools.cache
def _typeddict_fields(td: type) -> dict[str, tuple[Any, bool]]:
    """Get fields of a TypedDict with their types and required status.

    Results are cached because TypedDict type structures are static at
    runtime, and ``get_type_hints()`` is expensive on Python 3.10-3.12.

    Strips ``Required``/``NotRequired`` wrappers so downstream sees the
    actual type (e.g. ``Literal["response"]``, not ``Required[Literal["response"]]``).

    Returns:
        Dict mapping field name to ``(type_hint, is_required)``.
    """
    hints = get_type_hints(td, include_extras=True)
    required = getattr(td, "__required_keys__", set())
    optional = getattr(td, "__optional_keys__", set())
    result: dict[str, tuple[Any, bool]] = {}
    for name, tp in hints.items():
        inner = _strip_required(tp)
        if name in required:
            result[name] = (inner, True)
        elif name in optional:
            result[name] = (inner, False)
        else:
            # Default: assume required (total=True is the default)
            result[name] = (inner, True)
    return result


@functools.cache
def _dataclass_fields(dc: type) -> dict[str, tuple[Any, bool]]:
    """Get fields of a dataclass with their types and required status.

    Results are cached because dataclass type structures are static at
    runtime, and ``get_type_hints()`` is expensive on Python 3.10-3.12.

    Returns:
        Dict mapping field name to ``(type_hint, is_required)``.
    """
    hints = get_type_hints(dc, include_extras=True)
    result: dict[str, tuple[Any, bool]] = {}
    for f in dataclasses.fields(dc):
        tp = hints.get(f.name, f.type)
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        result[f.name] = (tp, not has_default)
    return result


def _join_path(base: str, key: str | int) -> str:
    """Join a path segment."""
    if isinstance(key, int):
        return f"{base}[{key}]" if base else f"[{key}]"
    return f"{base}.{key}" if base else key


def _type_name(tp: Any) -> str:
    """Human-readable name for a type."""
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        base = typing.get_args(tp)[0]
        return _type_name(base)
    if origin is Union:
        args = typing.get_args(tp)
        return " | ".join(_type_name(a) for a in args)
    if origin is typing.Literal:
        args = typing.get_args(tp)
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin is not None:
        base_name = getattr(origin, "__name__", str(origin))
        args = typing.get_args(tp)
        if args:
            args_str = ", ".join(_type_name(a) for a in args)
            return f"{base_name}[{args_str}]"
        return base_name
    if isinstance(tp, type):
        return tp.__name__
    if tp is type(None):
        return "None"
    return str(tp)


@functools.cache
def _find_discriminator(union_args: tuple[Any, ...]) -> str | None:
    """Find a shared Literal field that can discriminate union members.

    Checks all TypedDict members for a common field whose type is
    ``Literal[...]``.  Returns the field name if found, else None.

    Results are cached because union type structures are static.
    """
    td_args = [a for a in union_args if _is_typeddict(a)]
    if len(td_args) < 2:
        return None

    # Get all Literal field names from first TypedDict
    first_fields = _typeddict_fields(td_args[0])
    candidates: list[str] = []
    for name, (hint, _req) in first_fields.items():
        base, _ = _unwrap_annotated(hint)
        if typing.get_origin(base) is typing.Literal:
            candidates.append(name)

    for name in candidates:
        # Check that all other TypedDicts also have this field as Literal
        all_have = True
        for td in td_args[1:]:
            fields = _typeddict_fields(td)
            if name not in fields:
                all_have = False
                break
            base, _ = _unwrap_annotated(fields[name][0])
            if typing.get_origin(base) is not typing.Literal:
                all_have = False
                break
        if all_have:
            return name

    return None


# ── Coercion ──

_COERCIONS: dict[tuple[type, type], Callable[[Any], Any]] = {
    (str, int): int,
    (str, float): float,
    (int, float): float,
    (list, tuple): tuple,
    (tuple, list): list,
}


def _try_coerce(value: Any, target_type: type, coerce: bool) -> tuple[Any, bool]:
    """Attempt to coerce *value* to *target_type*.

    Returns:
        ``(coerced_value, True)`` on success, ``(value, False)`` on failure.
    """
    if not coerce:
        return value, False
    actual = type(value)
    converter = _COERCIONS.get((actual, target_type))
    if converter is None:
        return value, False
    try:
        return converter(value), True
    except (ValueError, TypeError):
        return value, False


# ── Core Validation Walker ──

# Mapping from Python type to expected isinstance types
_SIMPLE_TYPES: dict[type, type | tuple[type, ...]] = {
    str: str,
    int: int,
    float: (int, float),  # int is valid for float
    bool: bool,
    bytes: bytes,
}


def _validate_annotated(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
) -> Any:
    """Validate an ``Annotated[base, ...]`` type: check base then constraints."""
    base, constraints = _unwrap_annotated(tp)
    err_count_before = len(errors)
    value = _validate(value, base, path, errors, coerce)
    if len(errors) == err_count_before:
        for c in constraints:
            if isinstance(c, FieldValidator):
                try:
                    value = c.validate(value)
                except (ValueError, AssertionError) as e:
                    errors.append(
                        ErrorDetail(
                            path=path or "$",
                            expected=str(c),
                            actual=repr(value),
                            message=f"Validator '{c}' failed for value {value!r} at '{path or '$'}': {e}",
                        )
                    )
            elif not c.check(value):
                errors.append(
                    ErrorDetail(
                        path=path or "$",
                        expected=str(c),
                        actual=repr(value),
                        message=f"Constraint {c} failed for value {value!r} at '{path or '$'}'",
                    )
                )
    return value


def _validate_literal(
    value: Any,
    path: str,
    errors: list[ErrorDetail],
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``Literal[...]`` type."""
    if value not in args:
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=f"Literal[{', '.join(repr(a) for a in args)}]",
                actual=repr(value),
                message=f"Expected one of {args!r} at '{path or '$'}', got {value!r}",
            )
        )
    return value


def _try_discriminated(
    value: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    non_none_args: list[Any],
) -> tuple[Any, bool]:
    """Try to resolve a discriminated union via a shared Literal field.

    Returns:
        ``(result, True)`` if a discriminator matched, ``(value, False)`` otherwise.
    """
    disc_field = _find_discriminator(tuple(non_none_args))
    if disc_field is None or not isinstance(value, dict) or disc_field not in value:
        return value, False
    disc_val = value[disc_field]
    for candidate in non_none_args:
        if not _is_typeddict(candidate):
            continue
        fields = _typeddict_fields(candidate)
        if disc_field not in fields:
            continue
        base, _ = _unwrap_annotated(fields[disc_field][0])
        if typing.get_origin(base) is typing.Literal:
            if disc_val in typing.get_args(base):
                return _validate(value, candidate, path, errors, coerce), True
    return value, False


def _validate_union(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``Union`` / ``Optional`` type."""
    none_args = [a for a in args if a is type(None)]
    non_none_args = [a for a in args if a is not type(None)]

    if value is None:
        if none_args:
            return value
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=_type_name(tp),
                actual="None",
                message=f"Expected {_type_name(tp)} at '{path or '$'}', got None",
            )
        )
        return value

    # Try discriminated union for TypedDicts
    result, matched = _try_discriminated(value, path, errors, coerce, non_none_args)
    if matched:
        return result

    # Try each variant, pick the one with no errors
    for variant in non_none_args:
        test_errors: list[ErrorDetail] = []
        result = _validate(value, variant, path, test_errors, coerce)
        if not test_errors:
            return result

    errors.append(
        ErrorDetail(
            path=path or "$",
            expected=_type_name(tp),
            actual=type(value).__name__,
            message=f"Value at '{path or '$'}' does not match any variant of {_type_name(tp)}",
        )
    )
    return value


def _validate_struct_fields(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    fields: dict[str, tuple[Any, bool]],
) -> Any:
    """Validate a struct-like type (TypedDict or dataclass) against its fields."""
    if not isinstance(value, dict):
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=tp.__name__,
                actual=type(value).__name__,
                message=f"Expected dict for {tp.__name__} at '{path or '$'}', got {type(value).__name__}",
            )
        )
        return value

    err_before = len(errors)

    for name, (field_tp, required) in fields.items():
        if required and name not in value:
            errors.append(
                ErrorDetail(
                    path=_join_path(path, name),
                    expected=_type_name(field_tp),
                    actual="MISSING",
                    message=f"Missing required field '{name}' at '{path or '$'}'",
                )
            )

    for name, val in value.items():
        if name in fields:
            field_tp, _ = fields[name]
            _validate(val, field_tp, _join_path(path, name), errors, coerce)

    # Run model validators only if no field-level errors were added
    validators = _MODEL_VALIDATORS.get(tp)
    if validators and len(errors) == err_before:
        for validator in validators:
            try:
                result = validator(value)
                if result is not None:
                    value = result
            except (ValueError, AssertionError) as e:
                errors.append(
                    ErrorDetail(
                        path=path or "$",
                        expected=f"{tp.__name__} model validation",
                        actual=str(e),
                        message=f"Model validator failed for {tp.__name__} at '{path or '$'}': {e}",
                    )
                )

    return value


def _validate_list(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``list[X]`` type."""
    if not isinstance(value, list):
        coerced, ok = _try_coerce(value, list, coerce)
        if ok:
            value = coerced
        else:
            errors.append(
                ErrorDetail(
                    path=path or "$",
                    expected=_type_name(tp),
                    actual=type(value).__name__,
                    message=f"Expected list at '{path or '$'}', got {type(value).__name__}",
                )
            )
            return value
    if args:
        item_tp = args[0]
        for i, item in enumerate(value):
            _validate(item, item_tp, _join_path(path, i), errors, coerce)
    return value


def _validate_dict(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``dict[K, V]`` type."""
    if not isinstance(value, dict):
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=_type_name(tp),
                actual=type(value).__name__,
                message=f"Expected dict at '{path or '$'}', got {type(value).__name__}",
            )
        )
        return value
    if args and len(args) == 2:
        key_tp, val_tp = args
        for k, v in value.items():
            _validate(k, key_tp, _join_path(path, f"<key:{k!r}>"), errors, coerce)
            _validate(v, val_tp, _join_path(path, str(k)), errors, coerce)
    return value


def _validate_tuple(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``tuple[X, ...]`` or ``tuple[X, Y, Z]`` type."""
    if not isinstance(value, (tuple, list)):
        coerced, ok = _try_coerce(value, tuple, coerce)
        if ok:
            value = coerced
        else:
            errors.append(
                ErrorDetail(
                    path=path or "$",
                    expected=_type_name(tp),
                    actual=type(value).__name__,
                    message=f"Expected tuple at '{path or '$'}', got {type(value).__name__}",
                )
            )
            return value
    if args:
        if len(args) == 2 and args[1] is Ellipsis:
            item_tp = args[0]
            for i, item in enumerate(value):
                _validate(item, item_tp, _join_path(path, i), errors, coerce)
        elif len(value) != len(args):
            errors.append(
                ErrorDetail(
                    path=path or "$",
                    expected=f"tuple of length {len(args)}",
                    actual=f"length {len(value)}",
                    message=f"Expected tuple of {len(args)} elements at '{path or '$'}', got {len(value)}",
                )
            )
        else:
            for i, (item, item_tp) in enumerate(zip(value, args)):
                _validate(item, item_tp, _join_path(path, i), errors, coerce)
    return value


def _validate_set(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
    args: tuple[Any, ...],
) -> Any:
    """Validate a ``set[X]`` or ``frozenset[X]`` type."""
    if not isinstance(value, (set, frozenset, list)):
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=_type_name(tp),
                actual=type(value).__name__,
                message=f"Expected set at '{path or '$'}', got {type(value).__name__}",
            )
        )
        return value
    items = value if isinstance(value, (set, frozenset)) else value
    if args:
        item_tp = args[0]
        for i, item in enumerate(items):
            _validate(item, item_tp, _join_path(path, i), errors, coerce)
    return value


def _validate_bool(value: Any, path: str, errors: list[ErrorDetail]) -> Any:
    """Validate a ``bool`` type."""
    if not isinstance(value, bool):
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected="bool",
                actual=type(value).__name__,
                message=f"Expected bool at '{path or '$'}', got {type(value).__name__}",
            )
        )
    return value


def _validate_int(
    value: Any, path: str, errors: list[ErrorDetail], coerce: bool
) -> Any:
    """Validate an ``int`` type (rejects bools)."""
    if isinstance(value, bool) or not isinstance(value, int):
        coerced, ok = _try_coerce(value, int, coerce)
        if ok:
            return coerced
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected="int",
                actual=type(value).__name__,
                message=f"Expected int at '{path or '$'}', got {type(value).__name__}",
            )
        )
    return value


def _validate_float(
    value: Any, path: str, errors: list[ErrorDetail], coerce: bool
) -> Any:
    """Validate a ``float`` type (rejects bools, accepts ints)."""
    if isinstance(value, bool):
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected="float",
                actual="bool",
                message=f"Expected float at '{path or '$'}', got bool",
            )
        )
        return value
    if not isinstance(value, (int, float)):
        coerced, ok = _try_coerce(value, float, coerce)
        if ok:
            return coerced
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected="float",
                actual=type(value).__name__,
                message=f"Expected float at '{path or '$'}', got {type(value).__name__}",
            )
        )
    return value


def _validate_simple(
    value: Any,
    tp: type,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
) -> Any:
    """Validate a simple type (bool, int, float, str, bytes, etc.)."""
    if tp is bool:
        return _validate_bool(value, path, errors)
    if tp is int:
        return _validate_int(value, path, errors, coerce)
    if tp is float:
        return _validate_float(value, path, errors, coerce)

    # General isinstance check for str, bytes, etc.
    if not isinstance(value, tp):
        coerced, ok = _try_coerce(value, tp, coerce)
        if ok:
            return coerced
        errors.append(
            ErrorDetail(
                path=path or "$",
                expected=tp.__name__,
                actual=type(value).__name__,
                message=f"Expected {tp.__name__} at '{path or '$'}', got {type(value).__name__}",
            )
        )
    return value


def _validate(
    value: Any,
    tp: Any,
    path: str,
    errors: list[ErrorDetail],
    coerce: bool,
) -> Any:
    """Recursively validate *value* against type *tp*.

    Mutates *errors* in place.  Returns the (possibly coerced) value.
    Dispatches to per-type helpers for each supported annotation kind.
    """
    if tp is Any:
        return value

    if tp is None or tp is type(None):
        if value is not None:
            errors.append(
                ErrorDetail(
                    path=path or "$",
                    expected="None",
                    actual=type(value).__name__,
                    message=f"Expected None at '{path or '$'}', got {type(value).__name__}",
                )
            )
        return value

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Annotated:
        return _validate_annotated(value, tp, path, errors, coerce)
    if origin is typing.Literal:
        return _validate_literal(value, path, errors, args)
    if origin is Union:
        return _validate_union(value, tp, path, errors, coerce, args)
    if origin is list:
        return _validate_list(value, tp, path, errors, coerce, args)
    if origin is dict:
        return _validate_dict(value, tp, path, errors, coerce, args)
    if origin is tuple:
        return _validate_tuple(value, tp, path, errors, coerce, args)
    if origin in (set, frozenset):
        return _validate_set(value, tp, path, errors, coerce, args)

    if _is_typeddict(tp):
        return _validate_struct_fields(
            value, tp, path, errors, coerce, _typeddict_fields(tp)
        )
    if _is_dataclass_type(tp):
        return _validate_struct_fields(
            value, tp, path, errors, coerce, _dataclass_fields(tp)
        )

    if isinstance(tp, type):
        return _validate_simple(value, tp, path, errors, coerce)

    # Fallback — unknown type, skip validation
    return value


# ── JSON Schema Walker ──

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    bytes: "string",
}


def _schema_annotated(tp: Any) -> dict[str, Any]:
    """Generate JSON Schema for an ``Annotated[base, ...]`` type."""
    base, constraints = _unwrap_annotated(tp)
    schema = _type_to_schema(base)
    for c in constraints:
        kw = c.schema_kw()
        if schema.get("type") == "array" and hasattr(c, "schema_kw_array"):
            kw = c.schema_kw_array()
        schema.update(kw)
    return schema


def _schema_union(args: tuple[Any, ...]) -> dict[str, Any]:
    """Generate JSON Schema for a ``Union`` type."""
    none_args = [a for a in args if a is type(None)]
    non_none = [a for a in args if a is not type(None)]

    if len(non_none) == 1 and none_args:
        schema = _type_to_schema(non_none[0])
        if "type" in schema:
            schema["type"] = [schema["type"], "null"]
        else:
            schema = {"oneOf": [schema, {"type": "null"}]}
        return schema

    return {"oneOf": [_type_to_schema(a) for a in args]}


def _schema_struct(fields: dict[str, tuple[Any, bool]]) -> dict[str, Any]:
    """Generate JSON Schema for a struct-like type (TypedDict or dataclass)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, (field_tp, is_required) in fields.items():
        properties[name] = _type_to_schema(field_tp)
        if is_required:
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _schema_tuple(args: tuple[Any, ...]) -> dict[str, Any]:
    """Generate JSON Schema for a ``tuple`` type."""
    if args:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [_type_to_schema(a) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    return {"type": "array"}


def _schema_list(args: tuple[Any, ...]) -> dict[str, Any]:
    """Generate JSON Schema for a ``list[X]`` type."""
    schema: dict[str, Any] = {"type": "array"}
    if args:
        schema["items"] = _type_to_schema(args[0])
    return schema


def _schema_dict(args: tuple[Any, ...]) -> dict[str, Any]:
    """Generate JSON Schema for a ``dict[K, V]`` type."""
    schema: dict[str, Any] = {"type": "object"}
    if args and len(args) == 2:
        schema["additionalProperties"] = _type_to_schema(args[1])
    return schema


def _schema_set(args: tuple[Any, ...]) -> dict[str, Any]:
    """Generate JSON Schema for a ``set`` or ``frozenset`` type."""
    schema: dict[str, Any] = {"type": "array", "uniqueItems": True}
    if args:
        schema["items"] = _type_to_schema(args[0])
    return schema


def _type_to_schema(tp: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict.

    Dispatches to per-type schema helpers for each supported annotation kind.
    """
    if tp is Any:
        return {}
    if tp is None or tp is type(None):
        return {"type": "null"}

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Annotated:
        return _schema_annotated(tp)
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is Union:
        return _schema_union(args)
    if origin is list:
        return _schema_list(args)
    if origin is dict:
        return _schema_dict(args)
    if origin is tuple:
        return _schema_tuple(args)
    if origin in (set, frozenset):
        return _schema_set(args)
    if _is_typeddict(tp):
        return _schema_struct(_typeddict_fields(tp))
    if _is_dataclass_type(tp):
        return _schema_struct(_dataclass_fields(tp))
    if isinstance(tp, type) and tp in _PY_TO_JSON_TYPE:
        return {"type": _PY_TO_JSON_TYPE[tp]}
    return {}


# ── Public API ──


def validate(data: Any, tp: Any, *, coerce: bool = False) -> Any:
    """Validate *data* against type annotation *tp*.

    Args:
        data: The data to validate.
        tp: A TypedDict class, dataclass class, or any type annotation.
        coerce: If True, attempt type coercion (e.g. str to int).

    Returns:
        The validated (and possibly coerced) data.

    Raises:
        ValidationError: If validation fails, with all errors collected.
    """
    errors: list[ErrorDetail] = []
    result = _validate(data, tp, "", errors, coerce)
    if errors:
        raise ValidationError(errors)
    return result


def json_schema(tp: Any, *, title: str | None = None) -> dict[str, Any]:
    """Generate a JSON Schema dict from a type annotation.

    Args:
        tp: A TypedDict class, dataclass class, or any type annotation.
        title: Optional title for the schema root.

    Returns:
        A JSON Schema dict (draft 2020-12 compatible subset).
    """
    schema = _type_to_schema(tp)
    if title is not None:
        schema["title"] = title
    elif isinstance(tp, type) and hasattr(tp, "__name__"):
        schema["title"] = tp.__name__
    return schema
