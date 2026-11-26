import glob as g
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import MessageParam
from anthropic.types import ToolParam
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


WORKDIR = Path.cwd()

DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
]


def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


def check_permission(block) -> bool:
    # Gate 1: Hard deny
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    # Gate 2 + 3: Rule matching → User approval
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False

    return True


class PathEscapeError(ValueError):
  """Raised when a target path resolves outside WORKDIR."""


def safe_path(path: str | Path) -> Path:
    if path is None or str(path).strip() == "":
        raise ValueError("Path must not be empty")

    base = WORKDIR.resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (base / raw)
    resolved = candidate.resolve()

    if not resolved.is_relative_to(base):
        raise PathEscapeError(f"Path '{path}' escapes WORKDIR '{base}'")

    return resolved


def run_read(path, limit=None):
    lines = safe_path(path).read_text().splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)


def run_write(path, content):
    safe_path(path).write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def run_edit(path, old_text, new_text):
    text = safe_path(path).read_text()
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1))
    return f"Edited {path}"


def run_glob(pattern):
    return "\n".join(g.glob(pattern, root_dir=WORKDIR))


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return output.strip() or "(命令执行成功，无输出)"
    except Exception as e:
        return f"执行出错: {e}"


TOOL_HANDLERS = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
}


def agent_loop():
    user_input = input("please input:").strip()

    messages: list[MessageParam] = [{
        "role": "user",
        "content": user_input,
    }]

    TOOLS: list[ToolParam] = [
        {
            "name": "bash",
            "description": "Run a shell command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "需要读取的文件路径",
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "需要读取的文件路径",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "需要写入的文件路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "需要写入文件的内容",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "edit_file",
            "description": "Replace text in file once.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "需要写入的文件路径",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "需要替换的内容",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "需要写入文件的内容",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
        {
            "name": "glob",
            "description": "Find files by pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "匹配模式",
                    },
                },
                "required": ["pattern"],
            },
        },
    ]
    model = os.environ.get("ANTHROPIC_MODEL") or "claude-opus-5-5"
    while True:
        response = client.messages.create(
            model=model,
            system="you are assistant",
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        print(response.content)

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                if not check_permission(block):           # ← NEW
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Permission denied."
                    })
                    continue
                handler = TOOL_HANDLERS[block.name]    # lookup
                output = handler(**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    agent_loop()
