import os
import subprocess

from anthropic import Anthropic
from anthropic.types import MessageParam
from anthropic.types import ToolParam
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

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


def agent_loop():
    user_input = input("please input:").strip()
    print(repr(user_input))
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
                        "description": "要执行的完整 bash 命令字符串，例如 'ls -la' 或 'python3 script.py'",
                    },
                },
                "required": ["command"],
            },
        }
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

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "bash":
                cmd = str(block.input.get("command", ""))
                output = run_bash(cmd)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    agent_loop()
