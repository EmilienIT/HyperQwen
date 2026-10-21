"""Recover the append point of resumable self-distillation JSONL output. Stdlib only."""
import json


def load_completed(path):
    """Keep complete records, drop only a torn final record and separate a complete final line.

    Corruption before the last record is an error; silently skipping it would leave a
    JSONL file that capture.py cannot read. Nothing is changed in that case.
    """
    done = set()
    try:
        output = open(path, "rb+")
    except FileNotFoundError:
        return done
    with output:
        needs_newline = False
        while True:
            offset = output.tell()
            line = output.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                if output.read(1):
                    raise ValueError(f"{path}: invalid JSONL record at byte {offset}") from exc
                output.truncate(offset)
                needs_newline = False
                break
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"{path}: record at byte {offset} has no id")
            done.add(record["id"])
            needs_newline = not line.endswith(b"\n")
        if needs_newline:
            output.write(b"\n")
    return done
