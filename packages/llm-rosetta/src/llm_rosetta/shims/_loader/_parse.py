"""Parsers and validators for the blocks of a ``provider.yaml``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from llm_rosetta.formats.batch import DECLARABLE_DIALECTS, batch_spec_for
from llm_rosetta.types.ir.constraints import Range, RequestConstraints

from llm_rosetta.contracts.shim import (
    TYPE_PARAM_KEYS,
    BatchDeclaration,
    ReasoningCapability,
)
from ._constants import (
    _CONNECTION_KEYS,
    _REMOVED_LEAF_KEYS,
    logger,
)


def _parse_reasoning_cap(
    cfg: dict,
    *,
    base: ReasoningCapability | None = None,
) -> ReasoningCapability:
    """Parse a reasoning config dict into a :class:`ReasoningCapability`.

    When *base* is provided (model_overrides), unset fields inherit from it.
    """
    _d = base  # shorthand for fallback defaults

    # effort_range: YAML gives [floor, ceiling] list → tuple
    raw_range = cfg.get("effort_range", _d.effort_range if _d else None)
    effort_range = tuple(raw_range) if isinstance(raw_range, list) else raw_range

    return ReasoningCapability(
        thinking_modes=cfg.get("thinking_modes", _d.thinking_modes if _d else None),
        thinking_default=cfg.get(
            "thinking_default", _d.thinking_default if _d else None
        ),
        effort_field=cfg.get(
            "effort_field", _d.effort_field if _d else "reasoning_effort"
        ),
        effort_range=effort_range,
        budget_ratio=cfg.get("budget_ratio", _d.budget_ratio if _d else None),
        visibility_modes=cfg.get(
            "visibility_modes", _d.visibility_modes if _d else None
        ),
        replay_missing=cfg.get("replay_missing", _d.replay_missing if _d else "as_is"),
        unsigned_blocks=cfg.get(
            "unsigned_blocks", _d.unsigned_blocks if _d else "as_is"
        ),
    )


def _reject_removed_keys(cfg: dict[str, Any], yaml_path: Path) -> None:
    """Refuse the top-level keys an earlier release read from a shim.

    The flat connection and tool spellings were removed in 0.14.0.  The schema
    already refuses an unknown key, but the loader reads only the keys it
    knows, so a flat ``default_base_url:`` would otherwise be dropped without a
    word and the shim would load without its upstream.

    Raises:
        ValueError: Naming the file, each removed key and the nested key that
            replaces it.
    """
    replaced = {
        key: _REMOVED_LEAF_KEYS[key] for key in cfg if key in _REMOVED_LEAF_KEYS
    }
    replaced.update(
        {key: f"connection.{key}" for key in _CONNECTION_KEYS if key in cfg}
    )
    if replaced:
        hints = ", ".join(f"{key} -> {new}" for key, new in sorted(replaced.items()))
        raise ValueError(
            f"{yaml_path}: flat key(s) removed in 0.14.0; nest them under "
            f"'connection:' or 'tools:' ({hints})"
        )


def _resolve_shim_pair(cfg: dict[str, Any], yaml_path: Path) -> tuple[str, str]:
    """Return the canonical ``(format, model_type)`` a shim declares.

    A shim declares both ``model_type`` and ``format``.  The pair is validated
    against the converter registry so a shim cannot claim a format another
    category owns.

    Raises:
        ValueError: With a message naming the shim file and the expected
            formats.  A ``base:`` key (removed in 0.14.0) is refused with the
            ``format:`` / ``model_type:`` pair that replaces it.
    """
    # Imported lazily: the converter package imports shims.provider_shim at
    # module scope (for ReasoningCapability), so a module-level import here
    # would close the cycle while the shims package is initialising.  The
    # vocabulary module is data-only and safe to import at any time.
    from llm_rosetta import retired_names
    from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE, MODEL_TYPES

    if "base" in cfg:
        raise ValueError(
            f"{yaml_path}: 'base:' was removed in 0.14.0; declare "
            f"'format: <format>' and 'model_type: <type>' instead "
            f"(model_type is one of {', '.join(MODEL_TYPES)})"
        )

    declared_type = cfg.get("model_type")
    if isinstance(declared_type, str):
        message = retired_names.retired_name_message(
            declared_type, retired_names.MODEL_TYPE
        )
        if message is not None:
            raise ValueError(f"{yaml_path}: {message}")

    # The schema requires the category with the format: without
    # ``model_type`` the pair would silently default to ``llm``, so a future
    # embedding shim that forgot the field would load as a chat shim.
    if declared_type is None:
        raise ValueError(
            f"{yaml_path}: model_type is required (one of {', '.join(MODEL_TYPES)})"
        )

    raw_format = cfg.get("format")
    if not isinstance(raw_format, str):
        raise ValueError(f"{yaml_path}: a shim needs 'format'")

    fmt = raw_format
    if declared_type not in MODEL_TYPES:
        raise ValueError(
            f"{yaml_path}: unknown model_type {declared_type!r}; "
            f"expected one of {', '.join(MODEL_TYPES)}"
        )
    if fmt not in FORMATS_BY_TYPE[declared_type]:
        from llm_rosetta import retired_names

        message = retired_names.retired_name_message(fmt, retired_names.PROVIDER_TYPE)
        if message is not None:
            raise ValueError(f"{yaml_path}: {message}")
        raise ValueError(
            f"{yaml_path}: format {fmt!r} is not a {declared_type} format; "
            f"available: {', '.join(FORMATS_BY_TYPE[declared_type])}"
        )
    return fmt, declared_type


def _parse_connection_argv(
    raw: Any, shim_name: str, field: str = "acp_argv"
) -> tuple[str, ...] | None:
    """Read a shim's ``connection.<field>`` as an argv tuple.

    A string would be split on whitespace by a shell and a list of non-strings
    would fail somewhere inside the subprocess call; refusing both here puts
    the error beside the file that has it.

    Args:
        raw: The declared value: a list of strings, or ``None``.
        shim_name: The shim's name, for the message.
        field: The connection key being parsed, for the message.

    Returns:
        The argv, or ``None`` when the shim declares none.

    Raises:
        ValueError: The value is not a list of non-empty strings.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            f"{shim_name}: connection.{field} must be a non-empty list of "
            f"strings (e.g. ['agy', 'acp'])"
        )
    argv = tuple(str(part) for part in raw)
    if any(not part for part in argv):
        raise ValueError(f"{shim_name}: connection.{field} has an empty element")
    return argv


def _parse_connection_meta(raw: Any, shim_name: str) -> dict[str, Any] | None:
    """Read a shim's ``connection.acp_session_meta`` as a mapping.

    The value is forwarded as the request's ``_meta`` object, so it must
    already have the shape the agent expects.  Refusing a non-mapping here
    puts the error beside the file that has it rather than on the first turn,
    where a JSON-RPC error would name the agent instead of the shim.

    Args:
        raw: The declared value: a mapping, or ``None``.
        shim_name: The shim's name, for the message.

    Returns:
        A copy of the mapping, or ``None`` when the shim declares none.

    Raises:
        ValueError: The value is not a mapping.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{shim_name}: connection.acp_session_meta must be a mapping "
            f"(e.g. {{disableBuiltInTools: true}})"
        )
    return dict(raw)


def _parse_connection_env_unset(
    raw: Any, shim_name: str, field: str = "acp_env_unset"
) -> tuple[str, ...]:
    """Read a shim's ``connection.<field>`` as a tuple of names.

    Args:
        raw: The declared value: a list of non-empty strings, or ``None``.
        shim_name: The shim's name, for the message.
        field: The connection key being parsed, for the message.

    Returns:
        The names to remove, or an empty tuple when none are declared.

    Raises:
        ValueError: The value is not a list of non-empty strings.
    """
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            f"{shim_name}: connection.{field} must be a non-empty list "
            f"of environment variable names"
        )
    names = tuple(str(name) for name in raw)
    if any(not name for name in names):
        raise ValueError(f"{shim_name}: connection.{field} has an empty element")
    return names


def _parse_transport_options(raw: Any, shim_name: str) -> dict[str, Any] | None:
    """Read a shim's ``connection.transport_options`` as an object.

    Core does not know any transport's option names; the transport the lane
    names validates the content.  Only the shape is checked here.

    Args:
        raw: The declared value: a mapping, or ``None``.
        shim_name: The shim's name, for the message.

    Returns:
        A copy of the mapping, or ``None`` when the shim declares none.

    Raises:
        ValueError: The value is not a mapping.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{shim_name}: connection.transport_options must be a mapping")
    return dict(raw)


def _parse_connection_token_command(
    raw: Any, provider_dir: Path, name: str
) -> tuple[str, ...] | None:
    """Parse ``connection.token_command`` from a shim declaration.

    A relative argument that names a file in the shim's own directory is
    rewritten to its absolute path, so a shim can ship the exchange beside its
    ``provider.yaml``; anything else (``python3``, a flag, an absolute path) is
    left for the gateway, which resolves bare names on ``PATH``.

    A malformed declaration is ignored with a warning rather than rejected:
    the shim's other facts are still usable, and a broken command must not
    stop the gateway from loading an otherwise valid config.
    """
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(part, str) and part for part in raw)
    ):
        logger.warning(
            "Shim %s: connection.token_command must be a non-empty list of "
            "strings; ignoring it",
            name,
        )
        return None
    # Resolve every argument that names a file in the shim's directory — both
    # ``[token.py]`` and the more common ``[python3, token.py]`` ship a script
    # beside the declaration.  A bare binary name (``python3``) and any flag
    # are left untouched, since no such file exists here.
    argv = []
    for part in raw:
        candidate = provider_dir / part
        argv.append(
            str(candidate.resolve())
            if not Path(part).is_absolute() and candidate.is_file()
            else part
        )
    return tuple(argv)


def _parse_hint_msg(raw: Any, yaml_path: Path) -> dict[str, str] | None:
    """Normalise a shim's ``hint_msg`` to ``{language code: text}``.

    A bare string is the single-language form and is stored under ``en``.  A
    mapping keeps one entry per language.  Blank entries are dropped, and an
    entry whose key is not a string (YAML reads a bare ``1:`` as an integer)
    is dropped with a warning rather than re-keyed.  A declaration that
    leaves nothing readable is ``None``, so the panel shows no hint rather
    than an empty box.

    Args:
        raw: The ``hint_msg`` value as read from the YAML file.
        yaml_path: The declaring file, named in the warning.

    Returns:
        The normalised mapping, or ``None`` when nothing was declared.
    """
    if isinstance(raw, str):
        text = raw.strip()
        return {"en": text} if text else None
    if isinstance(raw, dict):
        hints: dict[str, str] = {}
        for lang, text in raw.items():
            if not isinstance(lang, str):
                logger.warning(
                    "Ignoring hint_msg entry %r in %s: its key is not a language code",
                    lang,
                    yaml_path,
                )
                continue
            if isinstance(text, str) and text.strip():
                hints[lang] = text.strip()
        return hints or None
    if raw is not None:
        logger.warning(
            "Ignoring hint_msg in %s: expected a string or a language mapping",
            yaml_path,
        )
    return None


def _value_matches(value: Any, annotation: Any) -> bool:
    """Whether *value* fits an IR field's annotation.

    Deliberately narrow: the annotations of ``GenerationConfig`` are plain
    ``int``, ``float``, ``bool``, ``str``, ``Literal[...]``, ``list[...]`` and
    ``dict[...]``, and an annotation this does not understand is accepted
    rather than guessed at.

    Args:
        value: The declared default.
        annotation: The IR field's annotation object.

    Returns:
        ``True`` when the value can be written into that field.
    """
    from typing import Literal, get_args, get_origin

    if annotation is float:
        # A float field accepts an int (``0`` for ``0.0``) but not a bool,
        # which Python would otherwise read as 1/0.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    origin = get_origin(annotation)
    if origin is Literal:
        return value in get_args(annotation)
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(annotation)
        return all(_value_matches(item, args[0]) for item in value) if args else True
    if origin is dict:
        return isinstance(value, dict)
    return True


def _expected(annotation: Any) -> str:
    """Describe an annotation for an error message."""
    from typing import Literal, get_args, get_origin

    if get_origin(annotation) is Literal:
        return " or ".join(repr(a) for a in get_args(annotation))
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return f"list of {_expected(args[0])}" if args else "list"
    if origin is dict:
        return "mapping"
    return getattr(annotation, "__name__", repr(annotation))


def _check_default_value(
    path: str, value: Any, annotations: dict[str, Any], yaml_path: Path
) -> None:
    """Refuse a default whose type the IR field cannot hold.

    The path check exists because a declaration that silently matches nothing
    is worse than one that fails; a value of the wrong type is the same
    failure one step later, so it is refused at load time too.

    Args:
        path: The IR path, already validated against the generation config.
        value: The declared default.
        annotations: ``GenerationConfig.__annotations__``.
        yaml_path: The file, for the message.

    Raises:
        ValueError: When the value does not fit the field's annotation.
    """
    annotation = annotations[path.partition(".")[2]]
    if isinstance(annotation, str):  # a stringised annotation says nothing
        return
    if not _value_matches(value, annotation):
        raise ValueError(
            f"{yaml_path}: request.defaults.{path} expects "
            f"{_expected(annotation)}, got {value!r}"
        )


def _parse_request_constraints(
    cfg: dict[str, Any], yaml_path: Path
) -> RequestConstraints:
    """Read the shim's ``request:`` section: defaults and value ranges.

    Keys are IR paths (``generation.max_tokens``), the same spelling the
    conversion warning uses as its subject.  A path the IR request cannot
    carry is refused at load time: a declaration that silently matches nothing
    is worse than one that fails, because the body it was meant to fix still
    goes out.

    Args:
        cfg: The parsed ``provider.yaml``.
        yaml_path: The file, for error messages.

    Returns:
        The declared constraints; an empty object when the section is absent.

    Raises:
        ValueError: On an unknown key, a malformed path, a non-numeric bound,
            or an unknown ``on_out_of_range`` policy.
    """
    from llm_rosetta.types.ir.configs import GenerationConfig

    raw = cfg.get("request")
    if raw is None:
        return RequestConstraints()
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: request: must be a mapping")

    unknown = set(raw) - {"defaults", "ranges"}
    if unknown:
        raise ValueError(
            f"{yaml_path}: request: has unknown key(s) "
            f"{', '.join(sorted(unknown))}; it accepts defaults and ranges"
        )

    generation_types = dict(GenerationConfig.__annotations__)
    generation_keys = set(generation_types)

    def _check_path(path: Any, where: str) -> None:
        if not isinstance(path, str) or path.count(".") != 1:
            raise ValueError(
                f"{where}: {path!r} is not a valid IR path; expected "
                f"'generation.<field>'"
            )
        block, _, field_name = path.partition(".")
        if block != "generation" or field_name not in generation_keys:
            raise ValueError(
                f"{where}: {path!r} is not a field of the IR generation "
                f"config; expected 'generation.<field>' with field one of "
                f"{', '.join(sorted(generation_keys))}"
            )

    defaults: dict[str, Any] = {}
    raw_defaults = raw.get("defaults") or {}
    if not isinstance(raw_defaults, dict):
        raise ValueError(f"{yaml_path}: request.defaults must be a mapping")
    for path, value in raw_defaults.items():
        _check_path(path, f"{yaml_path}: request.defaults")
        _check_default_value(path, value, generation_types, yaml_path)
        defaults[path] = value

    ranges: dict[str, Range] = {}
    raw_ranges = raw.get("ranges") or {}
    if not isinstance(raw_ranges, dict):
        raise ValueError(f"{yaml_path}: request.ranges must be a mapping")
    for path, spec in raw_ranges.items():
        _check_path(path, f"{yaml_path}: request.ranges")
        ranges[path] = _parse_range(f"request.ranges.{path}", spec, yaml_path)
    return RequestConstraints(defaults=defaults, ranges=ranges)


def _parse_range(where: str, spec: Any, yaml_path: Path) -> Range:
    """Parse one ``request.ranges`` entry into a :class:`Range`.

    ``on_out_of_range`` is required rather than defaulted: a range without a
    policy would clamp by surprise, and clamping changes a value the client
    sent.  The Python type keeps its default for programmatic declarations.

    Args:
        where: The entry's dotted location, for the message.
        spec: The YAML value.
        yaml_path: The file, for the message.

    Returns:
        The parsed range.

    Raises:
        ValueError: On a malformed entry.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"{yaml_path}: {where} must be a mapping with min, max and on_out_of_range"
        )
    unknown = set(spec) - {"min", "max", "on_out_of_range"}
    if unknown:
        raise ValueError(
            f"{yaml_path}: {where} has unknown key(s) {', '.join(sorted(unknown))}"
        )
    if "on_out_of_range" not in spec:
        raise ValueError(
            f"{yaml_path}: {where} must declare on_out_of_range (clamp, drop or reject)"
        )
    policy = spec["on_out_of_range"]
    if policy not in ("clamp", "drop", "reject"):
        raise ValueError(
            f"{yaml_path}: {where}.on_out_of_range must be clamp, drop or "
            f"reject, got {policy!r}"
        )
    bounds: dict[str, float | None] = {}
    for bound in ("min", "max"):
        value = spec.get(bound)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise ValueError(
                f"{yaml_path}: {where}.{bound} must be a number, got {value!r}"
            )
        bounds[bound] = value
    return Range(min=bounds["min"], max=bounds["max"], on_out_of_range=policy)


def _parse_type_params(
    cfg: dict[str, Any], model_type: str, yaml_path: Path
) -> dict[str, Any]:
    """Return the request-field defaults a shim declares for its own type.

    The section is named after the model type (``embeddings:``, ``rerank:``,
    ``decisions:``) and its keys are the IR field names the target converter
    already writes.  A section for another type is a mistake rather than a
    declaration, so it is refused, as is a key the type does not know.  The
    pre-rename spelling (``embedding:``) is refused with its replacement.

    Raises:
        ValueError: Naming the file, the offending section or key, and what
            the type accepts.
    """
    from llm_rosetta import retired_names

    values = dict(cfg)
    for retired in retired_names.RETIRED_NAMES:
        if (
            retired.kind == retired_names.MODEL_TYPE
            and retired.replacement == model_type
            and retired.name in values
        ):
            raise ValueError(
                f"{yaml_path}: "
                + retired_names.retired_key_message(
                    f"{retired.name}:", f"{model_type}:"
                )
            )
    declared = {key for key in TYPE_PARAM_KEYS if key in values}
    foreign = declared - {model_type}
    if foreign:
        raise ValueError(
            f"{yaml_path}: declares a {', '.join(sorted(foreign))} section but "
            f"the shim is {model_type}; only its own type's section is read"
        )
    raw = values.get(model_type)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{yaml_path}: {model_type} must be a mapping of parameter defaults"
        )
    allowed = TYPE_PARAM_KEYS[model_type]
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{yaml_path}: unknown {model_type} parameter(s) "
            f"{', '.join(sorted(unknown))}; allowed: "
            f"{', '.join(sorted(allowed)) or 'none'}"
        )
    for key, value in raw.items():
        if key in {"dimensions", "top_n", "max_tokens_per_doc"}:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{yaml_path}: {model_type}.{key} must be an integer")
        elif key in {"truncation", "return_documents"}:
            if not isinstance(value, bool):
                raise ValueError(f"{yaml_path}: {model_type}.{key} must be a boolean")
        elif not isinstance(value, str):
            raise ValueError(f"{yaml_path}: {model_type}.{key} must be a string")
    return dict(raw)


#: Keys a leaf's ``batch:`` block may declare, read off the declaration so
#: the two cannot drift.
_BATCH_KEYS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(BatchDeclaration)
)

_BATCH_PATH_KEYS: tuple[str, ...] = (
    "create_path",
    "retrieve_path",
    "cancel_path",
    "results_path",
)


def _parse_batch(
    cfg: dict[str, Any], model_type: str, fmt: str, yaml_path: Path
) -> BatchDeclaration | None:
    """Return the batch surface a leaf declares, or ``None`` for none.

    An absent ``batch:`` block is the normal case and means the lane has no
    batch API this repository declares — a shim written before batch existed
    keeps loading unchanged.  A present block is validated, because a typo
    must not read as "no batch": the lane would silently answer
    ``unsupported`` for a provider that does support batching.

    The declared dialect must be one this shim's own ``(model_type, format)``
    batch record offers (``emulated`` is allowed for any measured format), so
    a messages lane cannot name the OpenAI file codec by mistake.

    Raises:
        ValueError: The block is not a mapping, carries an unknown key, a
            non-string path, a non-boolean ``supports_files``, or a dialect other than
            ``emulated`` that the pair does not offer (a format with no batch
            record offers none, so only ``emulated`` is accepted there).
    """
    raw = cfg.get("batch")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: batch must be a mapping")
    unknown = set(raw) - set(_BATCH_KEYS)
    if unknown:
        raise ValueError(
            f"{yaml_path}: unknown batch key(s) {', '.join(sorted(unknown))}; "
            f"a batch declaration may carry {', '.join(_BATCH_KEYS)}"
        )
    dialect = raw.get("dialect")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError(
            f"{yaml_path}: batch must name a dialect (one of "
            f"{', '.join(DECLARABLE_DIALECTS)})"
        )
    spec = batch_spec_for(model_type, fmt)
    offered: tuple[str, ...] = ()
    if spec is not None:
        offered = (spec.dialect, *((spec.file_dialect,) if spec.file_dialect else ()))
    if dialect != "emulated" and dialect not in offered:
        raise ValueError(
            f"{yaml_path}: batch.dialect {dialect!r} is not offered by "
            f"{model_type}/{fmt}"
            + (f" (offered: {', '.join(offered)}, or emulated)" if offered else "")
        )
    supports_files = raw.get("supports_files", False)
    if not isinstance(supports_files, bool):
        raise ValueError(f"{yaml_path}: batch.supports_files must be a boolean")
    paths: dict[str, str | None] = {}
    for key in _BATCH_PATH_KEYS:
        value = raw.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{yaml_path}: batch.{key} must be a path string")
        paths[key] = value
    return BatchDeclaration(
        dialect=dialect,
        supports_files=supports_files,
        **paths,
    )
