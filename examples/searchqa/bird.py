"""SQL extraction and execution accuracy for the fixed BIRD split."""

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


DEFAULT_DB_ROOT = Path(__file__).resolve().parents[3] / "bird-text-to-sql-260820/data_dev/dev_databases"


def database_path(root: Path, db_id: str) -> Path:
    if not db_id or Path(db_id).name != db_id or db_id in {".", ".."}:
        raise ValueError(f"Invalid BIRD database ID: {db_id!r}")
    return root / db_id / f"{db_id}.sqlite"


def extract_sql(content: str) -> str:
    if not content:
        return ""
    content = content.rsplit(" response", 1)[-1]
    fenced = re.search(r"```(?:sql|sqlite|mysql|postgresql)?\s*([\s\S]*?)```", content, re.IGNORECASE)
    if fenced:
        content = fenced.group(1)
    start = re.search(r"(?im)^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b", content)
    if start:
        content = content[start.start():]
    content = content.strip()
    # The first unquoted semicolon ends the statement. SQLite doubles quote
    # characters inside literals and identifiers.
    quote_char = None
    i = 0
    while i < len(content):
        ch = content[i]
        if quote_char:
            if ch == quote_char:
                if i + 1 < len(content) and content[i + 1] == quote_char:
                    i += 2
                    continue
                quote_char = None
        elif ch in ("'", '"', "`"):
            quote_char = ch
        elif ch == ";":
            return content[:i + 1]
        i += 1
    return content


def _read_only(action, arg1, arg2, db, trigger):
    # A read-only main DB alone does not stop ATTACH from creating another file.
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() not in {
        "table_info", "table_xinfo", "table_list", "index_info", "index_xinfo",
        "index_list", "foreign_key_list", "database_list", "compile_options",
    }:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ("load_extension", "writefile"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def evaluate_sql(response: str, gold_sql: str, path: Path, timeout: float = 30.0) -> dict:
    predicted_sql = extract_sql(response)
    result = {"predicted_sql": predicted_sql, "ex": 0, "exec_status": "error", "error": ""}
    if not predicted_sql:
        result["error"] = "empty SQL"
        return result
    deadline = time.monotonic() + timeout
    expired = False

    def interrupt():
        nonlocal expired
        expired = time.monotonic() > deadline
        return int(expired)

    stage = "connect"
    try:
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(_read_only)
            conn.set_progress_handler(interrupt, 1000)
            stage = "predicted SQL"
            predicted = set(conn.execute(predicted_sql).fetchall())
            stage = "gold SQL"
            gold = set(conn.execute(gold_sql).fetchall())
        result.update(ex=int(predicted == gold), exec_status="ok")
    except sqlite3.Error as exc:
        if expired:
            result.update(exec_status="timeout", error=f"timeout after {timeout:g}s ({stage})")
        else:
            result["error"] = f"{stage}: {type(exc).__name__}: {exc}"
    return result
