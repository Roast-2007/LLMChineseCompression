"""End-to-end round-trip tests for compress/decompress."""

import sys
sys.path.insert(0, "src")

from zippedtext.compressor import compress, decompress


def test_chinese_text():
    text = "深度学习是人工智能的核心技术，它通过神经网络模拟人脑的工作方式。"
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text, f"Mismatch:\n  orig: {text}\n  got:  {restored}"
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[Chinese] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


def test_english_text():
    text = "The quick brown fox jumps over the lazy dog. Machine learning is transforming the world."
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[English] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


def test_digits():
    text = "3141592653589793238462643383279502884197"
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[Digits] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


def test_mixed_content():
    text = "2024年，OpenAI发布了GPT-4模型，参数量达到1.8万亿。中国的DeepSeek也推出了V3模型。"
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[Mixed] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


def test_longer_chinese():
    text = (
        "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。"
        "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。"
        "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"
        "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，"
        "所以它与语言学的研究有着密切的联系。自然语言处理技术主要包括"
        "文本分类、信息抽取、机器翻译、问答系统、对话系统等。"
        "近年来，随着深度学习技术的发展，自然语言处理取得了巨大的进步。"
        "特别是Transformer架构的提出，使得预训练语言模型成为可能。"
    )
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[Long CN] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


def test_single_char():
    text = "好"
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    print(f"[Single] OK")


def test_repeated_text():
    text = "你好" * 100
    data = compress(text, mode="offline")
    restored = decompress(data)
    assert restored == text
    ratio = len(data) / len(text.encode("utf-8"))
    print(f"[Repeat] {len(text)}chars, {len(text.encode('utf-8'))}B → {len(data)}B, ratio={ratio:.2f}")


if __name__ == "__main__":
    test_chinese_text()
    test_english_text()
    test_digits()
    test_mixed_content()
    test_longer_chinese()
    test_single_char()
    test_repeated_text()
    print("\nAll round-trip tests passed!")
