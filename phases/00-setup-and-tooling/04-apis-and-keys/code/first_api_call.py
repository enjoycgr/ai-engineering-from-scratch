# Phase 0 · Lesson 04 — API 与密钥。
# 演示使用 SDK 和原始 HTTP 两种方式调用 Anthropic API。
# Refs: https://docs.anthropic.com/en/api/messages

import os
import json
import urllib.request


def call_with_sdk():
    """使用 Anthropic Python SDK 调用 API。"""
    try:
        import anthropic
    except ImportError:
        print("请安装 SDK: pip install anthropic")
        return

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
    )
    print(f"SDK 响应: {response.content[0].text}")
    print(f"使用 token 数: {response.usage.input_tokens} 输入, {response.usage.output_tokens} 输出")


def call_raw_http():
    """使用原始 HTTP 请求调用 Anthropic API（不使用 SDK）。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("请先设置 ANTHROPIC_API_KEY 环境变量")
        return

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
    }).encode()

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"原始 HTTP 响应: {result['content'][0]['text']}")
        print(f"使用 token 数: {result['usage']['input_tokens']} 输入, {result['usage']['output_tokens']} 输出")


if __name__ == "__main__":
    print("=== API 调用 ===\n")
    print("1. 使用 SDK:")
    call_with_sdk()
    print("\n2. 使用原始 HTTP:")
    call_raw_http()
