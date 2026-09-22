# -*- coding: utf-8 -*-
"""
现成知识库加工脚本（chapter09 数据处理思路）
------------------------------------------------------------------
数据来源（非手写，来自 HuggingFace 开源数据集，国内镜像可直接下载）：
    sunorme/chinese-adorable-high-emotional-intelligence-chat
    https://hf-mirror.com/datasets/sunorme/chinese-adorable-high-emotional-intelligence-chat
    170 条中文高情商恋爱日常对话，字段 user / girl

本脚本把原始 JSON 对话对加工成 app.py 的 RAG 数据源 knowledge.txt：
    【高情商话术】对方说"<user>"，可以这样回应："<girl>"
每条占一行、条目间空行分隔，保持与原手写知识库相同的【标签】格式，
RecursiveCharacterTextSplitter 可按段落完整切分。

重新生成：
    python build_knowledge.py
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "raw_data", "chinese-adorable-high-eq-chat.json")
OUT_FILE = os.path.join(BASE_DIR, "knowledge.txt")


def main():
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    entries = []
    seen = set()  # 按对话内容去重
    for row in rows:
        user = row.get("user", "").strip().replace("\n", " ")
        girl = row.get("girl", "").strip().replace("\n", " ")
        if not user or not girl:
            continue
        key = (user, girl)
        if key in seen:
            continue
        seen.add(key)
        entries.append(f'【高情商话术】对方说"{user}"，可以这样回应："{girl}"')

    # 每条之间空一行，与原 knowledge.txt 的排版一致
    content = "\n\n".join(entries) + "\n"
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    max_len = max(len(e) for e in entries)
    print(f"已生成 {OUT_FILE}")
    print(f"知识条目: {len(entries)} 条（原始 {len(rows)} 条）| 最长条目 {max_len} 字")


if __name__ == "__main__":
    main()
