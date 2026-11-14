"""A tiny, readable condition language for early-game rules, with three-valued evaluation.

    our.jng.lvl1 == Strong AND their.jng.lvl1 != Strong
    our.timing == Early AND (our.bot.prio == Strong OR our.mid.prio == Strong)
    our.{lane}.blind AND our.{lane}.matchup_dependent          ({lane} is filled in by the rule's for_each)
    their.sup.style == "Protect the carry"
    ALWAYS

Grammar: OR / AND / NOT, parentheses, comparisons == != < <= > >= IN [a, b], a bare fact means "is true".
Values are words, "quoted strings" or numbers. parse() turns the text into a JSON-able tree; evaluate() returns
True / False / None (unknown: a fact it needs has no value yet) plus the facts that were missing.
"""
from __future__ import annotations

import re

from .facts import fact_spec

TOKEN = re.compile(r'\s*(?:(\()|(\))|(\[)|(\])|(,)|(==|!=|<=|>=|<|>)|"([^"]*)"|([A-Za-z_][\w.{}/+-]*|-?\d+(?:\.\d+)?))')
KEYWORDS = {"AND", "OR", "NOT", "IN", "ALWAYS"}


class ConditionError(ValueError):
    pass


def _tokens(text):
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ConditionError(f"can't read condition at: {text[pos:pos + 25]!r}")
        lp, rp, lb, rb, comma, op, quoted, word = m.groups()
        if quoted is not None:
            out.append(("str", quoted))
        elif word is not None:
            out.append(("kw", word.upper()) if word.upper() in KEYWORDS else ("word", word))
        else:
            out.append(("sym", lp or rp or lb or rb or comma or op))
        pos = m.end()
    return out


def _value(tok):
    kind, v = tok
    if kind == "str":
        return v
    if v.startswith(("our.", "their.")):
        return {"ref": v}            # compare with another fact, e.g. our.x > their.x
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def parse(text: str) -> dict:
    toks = _tokens(text)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take(expect=None):
        nonlocal i
        if i >= len(toks):
            raise ConditionError(f"condition ends too early: {text!r}")
        t = toks[i]
        if expect and t[1] != expect:
            raise ConditionError(f"expected {expect!r} but found {t[1]!r} in {text!r}")
        i += 1
        return t

    def p_or():
        parts = [p_and()]
        while peek() == ("kw", "OR"):
            take()
            parts.append(p_and())
        return parts[0] if len(parts) == 1 else {"or": parts}

    def p_and():
        parts = [p_unary()]
        while peek() == ("kw", "AND"):
            take()
            parts.append(p_unary())
        return parts[0] if len(parts) == 1 else {"and": parts}

    def p_unary():
        t = peek()
        if t == ("kw", "NOT"):
            take()
            return {"not": p_unary()}
        if t == ("kw", "ALWAYS"):
            take()
            return {"always": True}
        if t == ("sym", "("):
            take()
            e = p_or()
            take(")")
            return e
        kind, name = take()
        if kind != "word":
            raise ConditionError(f"expected a fact name, found {name!r} in {text!r}")
        nxt = peek()
        if nxt[0] == "sym" and nxt[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = take()[1]
            return {"fact": name, "op": op, "value": _value(take())}
        if nxt == ("kw", "IN"):
            take()
            take("[")
            vals = [_value(take())]
            while peek() == ("sym", ","):
                take()
                vals.append(_value(take()))
            take("]")
            return {"fact": name, "op": "in", "value": vals}
        return {"fact": name, "op": "is", "value": True}

    tree = p_or()
    if i != len(toks):
        raise ConditionError(f"unexpected {toks[i][1]!r} in {text!r}")
    return tree


def facts_used(tree) -> list:
    if "fact" in tree:
        v = tree["value"]
        return [tree["fact"]] + ([v["ref"]] if isinstance(v, dict) else [])
    if "not" in tree:
        return facts_used(tree["not"])
    out = []
    for k in ("and", "or"):
        for t in tree.get(k, []):
            out += facts_used(t)
    return out


def validate(tree, placeholders: dict | None = None) -> list:
    """Problems with a parsed condition: unknown fact names or values outside a fact's allowed set."""
    problems = []
    for name in set(facts_used(tree)):
        names = [name]
        for ph, vals in (placeholders or {}).items():
            if "{" + ph + "}" in name:
                names = [name.replace("{" + ph + "}", v) for v in vals]
        for n in names:
            if fact_spec(n) is None:
                problems.append(f"unknown fact {n!r}")
    for node in _comparisons(tree):
        spec = fact_spec(_fill(node["fact"], placeholders))
        if isinstance(node["value"], dict):
            other = fact_spec(_fill(node["value"]["ref"], placeholders))
            if spec and other and spec[0] != other[0]:
                problems.append(f"{node['fact']} ({spec[0]}) compared with {node['value']['ref']} ({other[0]})")
            continue
        if spec and spec[1] and node["op"] in ("==", "!=", "in"):
            vals = node["value"] if isinstance(node["value"], list) else [node["value"]]
            bad = [v for v in vals if v not in spec[1]]
            if bad:
                problems.append(f"{node['fact']}: {bad} not in {spec[1]}")
    return problems


def _fill(name, placeholders):
    for ph, vals in (placeholders or {}).items():
        name = name.replace("{" + ph + "}", vals[0])
    return name


def _comparisons(tree):
    if "fact" in tree:
        yield tree
    elif "not" in tree:
        yield from _comparisons(tree["not"])
    else:
        for k in ("and", "or"):
            for t in tree.get(k, []):
                yield from _comparisons(t)


def substitute(tree, mapping: dict):
    """Copy of the tree with {placeholder}s in fact names replaced."""
    if "fact" in tree:
        name = tree["fact"]
        val = tree["value"]
        for k, v in mapping.items():
            name = name.replace("{" + k + "}", v)
            if isinstance(val, dict):
                val = {"ref": val["ref"].replace("{" + k + "}", v)}
        return {**tree, "fact": name, "value": val}
    if "not" in tree:
        return {"not": substitute(tree["not"], mapping)}
    for k in ("and", "or"):
        if k in tree:
            return {k: [substitute(t, mapping) for t in tree[k]]}
    return dict(tree)


def _ord(fact, v):
    spec = fact_spec(fact)
    if spec and spec[1] and v in spec[1]:
        return spec[1].index(v)
    return v


def evaluate(tree, facts: dict):
    """(True | False | None, missing facts). None = unknown because a needed fact has no value (Kleene logic)."""
    missing = []

    def ev(t):
        if t.get("always"):
            return True
        if "not" in t:
            r = ev(t["not"])
            return None if r is None else not r
        if "and" in t:
            rs = [ev(x) for x in t["and"]]
            return False if False in rs else (None if None in rs else True)
        if "or" in t:
            rs = [ev(x) for x in t["or"]]
            return True if True in rs else (None if None in rs else False)
        name, op, val = t["fact"], t["op"], t["value"]
        v = facts.get(name)
        if v is None:
            missing.append(name)
            return None
        if isinstance(val, dict):
            val = facts.get(val["ref"])
            if val is None:
                missing.append(t["value"]["ref"])
                return None
            a, b = _ord(name, v), _ord(t["value"]["ref"], val)
            try:
                return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
            except TypeError:
                return False
        if op == "is":
            return bool(v)
        if op == "in":
            return v in val
        a, b = _ord(name, v), _ord(name, val)
        try:
            return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        except TypeError:
            return False

    return ev(tree), sorted(set(missing))


def plain(tree) -> str:
    """Plain-language rendering for coaches, e.g. "our jungler's level-1 strength is Strong and ..."."""
    if tree.get("always"):
        return "always"
    if "not" in tree:
        return "not (" + plain(tree["not"]) + ")"
    for k, word in (("and", " and "), ("or", " or ")):
        if k in tree:
            return word.join(("(" + plain(t) + ")") if ("and" in t or "or" in t) else plain(t) for t in tree[k])
    spec = fact_spec(tree["fact"])
    desc = spec[3] if spec else tree["fact"]
    op, val = tree["op"], tree["value"]
    if op == "is":
        return desc
    if isinstance(val, dict):
        ref = fact_spec(val["ref"])
        val = ref[3] if ref else val["ref"]
    words = {"==": "is", "!=": "is not", "<": "is below", "<=": "is at most", ">": "is above", ">=": "is at least",
             "in": "is one of"}[op]
    return f"{desc} {words} {', '.join(map(str, val)) if isinstance(val, list) else val}"
