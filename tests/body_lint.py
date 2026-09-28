"""Find the names a function body reaches through search_path (issue #30).

A plpgsql body is resolved statement by statement as it runs, so creating it
proves nothing about the names it uses, and a body that is exercised by no test
could still name `chunks` where it means `pgkg_host.chunks`.  This reads the
body instead: tokenised as PostgreSQL would, with comments and ordinary string
literals set aside, and with the strings that are SQL — the ones handed to
EXECUTE or format(), and dollar-quoted blocks — read as code too.

What counts as a pgkg or extension name comes from the catalog (`Names.from_
catalog`), not from a list kept here, so an object a later migration adds is
checked without anyone remembering to add it.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import asyncpg

_TOKEN = re.compile(
    r"""(?P<ws>\s+)
    |(?P<line_comment>--[^\n]*)
    |(?P<block_comment>/\*.*?\*/)
    |(?P<estr>[eE]'(?:[^'\\]|\\.|'')*')
    |(?P<str>'(?:[^']|'')*')
    |(?P<dollar>\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$)
    |(?P<param>\$\d+)
    |(?P<qid>"(?:[^"]|"")*")
    |(?P<cast>::)
    |(?P<id>[A-Za-z_][A-Za-z0-9_$]*)
    |(?P<num>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    |(?P<op>[+\-*/<>=~!@#%^&|`?]+)
    |(?P<ch>.)""",
    re.S | re.X,
)

# format()'s conversions, which would otherwise read as the `%` operator.
_FORMAT_CONVERSION = re.compile(r"%(?:\d+\$)?-?\d*[sIL%]")

_REG_TYPES = frozenset(
    {"regclass", "regproc", "regprocedure", "regtype", "regoper", "regoperator"}
)
_TO_REG_FUNCTIONS = frozenset(
    {"to_regclass", "to_regproc", "to_regprocedure", "to_regtype", "to_regoper"}
)
# The extensions overload these for their own types (vector = vector,
# vector + vector), and text cannot tell that use from the ordinary one, which
# is in every body.  A SQL body comparing vectors is still caught exactly, by
# re-creating it under an empty path; a plpgsql one is not.
_ORDINARY_OPERATORS = frozenset({"=", "<>", "<", "<=", ">", ">=", "+", "-", "*", "/", "||"})

_RELATION_POSITION = frozenset(
    {"from", "join", "into", "update", "table", "only", "references", "truncate"}
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    tag: str = ""

    @property
    def word(self) -> str:
        return self.text.lower() if self.kind == "id" else ""


def _tokens(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        assert match is not None
        kind = match.lastgroup or "ch"
        if kind == "dollar":
            tag = match.group(0)
            end = text.find(tag, match.end())
            if end < 0:
                tokens.append(_Token("ch", tag))
                position = match.end()
                continue
            tokens.append(_Token("dollar", text[position:end + len(tag)], tag))
            position = end + len(tag)
            continue
        if kind not in ("ws", "line_comment", "block_comment"):
            tokens.append(_Token(kind, match.group(0)))
        position = match.end()
    return tokens


class _Queryable(Protocol):
    async def fetch(self, query: str, *args: object) -> list[asyncpg.Record]: ...


@dataclass(frozen=True)
class Names:
    """What an unqualified name in a body could resolve to, by kind."""

    relations: frozenset[str]
    # Relation names that are also column or parameter names, so only a
    # relation position (FROM, JOIN, INTO, ...) makes them a relation.
    also_columns: frozenset[str]
    functions: frozenset[str]
    types: frozenset[str]
    opclasses: frozenset[str]
    # Operators only an extension defines, and ones pg_catalog defines too
    # (`%` is modulo as well as pg_trgm's similarity).
    extension_operators: frozenset[str]
    shared_operators: frozenset[str]

    @classmethod
    async def from_catalog(
        cls, conn: _Queryable, *, schema: str, extension_schemas: Iterable[str]
    ) -> Names:
        extension_schemas = sorted(set(extension_schemas) - {"pg_catalog"})

        async def column(sql: str, *args: object) -> frozenset[str]:
            return frozenset(row[0] for row in await conn.fetch(sql, *args))

        relations = await column(
            """
            SELECT c.relname FROM pg_catalog.pg_class c
            WHERE c.relnamespace = $1::pg_catalog.regnamespace
              AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'c', 'f')
            """,
            schema,
        )
        columns = await column(
            """
            SELECT a.attname FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
            WHERE c.relnamespace = $1::pg_catalog.regnamespace AND a.attnum > 0
            UNION
            SELECT pg_catalog.unnest(p.proargnames) FROM pg_catalog.pg_proc p
            WHERE p.pronamespace = $1::pg_catalog.regnamespace
            """,
            schema,
        )
        builtin_functions = await column(
            "SELECT proname FROM pg_catalog.pg_proc"
            " WHERE pronamespace = 'pg_catalog'::pg_catalog.regnamespace"
        )
        builtin_operators = await column(
            "SELECT oprname FROM pg_catalog.pg_operator"
            " WHERE oprnamespace = 'pg_catalog'::pg_catalog.regnamespace"
        )
        types = await column(
            """
            SELECT t.typname FROM pg_catalog.pg_type t
            JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
            WHERE n.nspname = ANY($1::TEXT[]) AND t.typelem = 0
            """,
            extension_schemas,
        )
        own_functions = await column(
            "SELECT proname FROM pg_catalog.pg_proc"
            " WHERE pronamespace = $1::pg_catalog.regnamespace",
            schema,
        )
        extension_functions = await column(
            """
            SELECT p.proname FROM pg_catalog.pg_proc p
            JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = ANY($1::TEXT[])
            """,
            extension_schemas,
        )
        opclasses = await column(
            """
            SELECT o.opcname FROM pg_catalog.pg_opclass o
            JOIN pg_catalog.pg_namespace n ON n.oid = o.opcnamespace
            WHERE n.nspname = ANY($1::TEXT[])
            """,
            extension_schemas,
        )
        operators = await column(
            """
            SELECT o.oprname FROM pg_catalog.pg_operator o
            JOIN pg_catalog.pg_namespace n ON n.oid = o.oprnamespace
            WHERE n.nspname = ANY($1::TEXT[])
            """,
            extension_schemas,
        )
        return cls(
            relations=relations,
            also_columns=relations & columns,
            functions=(own_functions | (extension_functions - builtin_functions)) - types,
            types=types,
            opclasses=opclasses,
            extension_operators=operators - builtin_operators,
            shared_operators=(operators & builtin_operators) - _ORDINARY_OPERATORS,
        )


def unqualified_references(body: str, names: Names) -> list[str]:
    """Every name in `body` that only search_path could resolve, described."""
    return _scan(_tokens(body), names)


def _cte_names(tokens: Sequence[_Token]) -> frozenset[str]:
    found: set[str] = set()
    for i, token in enumerate(tokens[:-2]):
        if token.kind != "id" or tokens[i + 1].word != "as":
            continue
        after = tokens[i + 2]
        if after.text == "(" or after.word in ("materialized", "not"):
            found.add(token.word)
    return frozenset(found)


def _string_value(token: _Token) -> str:
    if token.kind == "dollar":
        return token.text[len(token.tag):-len(token.tag)]
    literal = token.text[1:] if token.kind == "estr" else token.text
    return literal[1:-1].replace("''", "'")


def _scan(tokens: Sequence[_Token], names: Names) -> list[str]:
    found: list[str] = []
    ctes = _cte_names(tokens)
    openers: list[str] = []
    execute_depth: int | None = None

    def at(i: int) -> _Token:
        return tokens[i] if 0 <= i < len(tokens) else _Token("", "")

    for i, token in enumerate(tokens):
        prev, nxt = at(i - 1), at(i + 1)

        if token.text == "(":
            openers.append(prev.word)
            continue
        if token.text == ")":
            if openers:
                openers.pop()
            continue
        if token.text == ";" or (
            execute_depth == len(openers) and token.word in ("using", "into")
        ):
            execute_depth = None
            continue
        if token.word == "execute" and nxt.word not in ("function", "procedure"):
            execute_depth = len(openers)
            continue

        if token.kind in ("str", "estr", "dollar"):
            value = _string_value(token)
            in_format = bool(openers) and openers[-1] == "format"
            if token.kind == "dollar" or in_format or execute_depth is not None:
                code = _FORMAT_CONVERSION.sub(" __fmt__ ", value)
                found += _scan(_tokens(code), names)
                continue
            is_reg_cast = nxt.kind == "cast" and at(i + 2).word in _REG_TYPES
            in_to_reg = bool(openers) and openers[-1] in _TO_REG_FUNCTIONS
            if (is_reg_cast or in_to_reg) and "." not in value:
                target = value.split("(")[0].strip().lower()
                if target in names.relations | names.functions | names.types:
                    found.append(f"name resolved at run time: {token.text}")
            continue

        if token.kind == "op":
            if prev.text == ".":
                continue
            if token.text in names.extension_operators:
                found.append(f"operator {token.text}")
            elif token.text in names.shared_operators:
                if nxt.kind == "num" or nxt.word in ("type", "rowtype"):
                    continue
                found.append(f"operator {token.text}")
            continue

        if token.kind != "id" or prev.text == ".":
            continue
        word = token.word

        if nxt.text == ".":
            is_column_type = (
                at(i + 2).kind == "id"
                and at(i + 3).text == "%"
                and at(i + 4).word in ("type", "rowtype")
            )
            if is_column_type and word in names.relations:
                found.append(f"relation {word} in %TYPE")
            continue
        if nxt.text == "%" and at(i + 2).word in ("type", "rowtype"):
            if word in names.relations:
                found.append(f"relation {word} in %ROWTYPE")
            continue
        if word in names.functions and nxt.text == "(":
            found.append(f"function {word}()")
            continue
        if word in names.types:
            found.append(f"type {word}")
            continue
        if word in names.opclasses:
            found.append(f"operator class {word}")
            continue
        if word in names.relations and word not in ctes:
            aliased = prev.word == "as" or (prev.kind == "id" and at(i - 2).text == ".")
            if aliased:
                continue
            if word in names.also_columns and prev.word not in _RELATION_POSITION:
                continue
            found.append(f"relation {word}")
    return found
