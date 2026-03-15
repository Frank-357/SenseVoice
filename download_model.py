#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性预下载 SenseVoice 模型到本地缓存。

用法：
    python download_model.py
    python download_model.py --model "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="预下载 SenseVoice 模型")
    parser.add_argument(
        "--model",
        default="iic/SenseVoiceSmall",
        help="ModelScope 模型 ID（默认：iic/SenseVoiceSmall）",
    )
    args = parser.parse_args()

    print(f"正在下载模型：{args.model}")
    print("首次下载约 230MB（SenseVoiceSmall），请耐心等待...\n")

    try:
        from model import SenseVoiceSmall

        m, kwargs = SenseVoiceSmall.from_pretrained(model=args.model, device="cpu")
        print(f"\n模型已下载到：{kwargs['model_path']}")
        print("下载完成。可以运行 hotkey_daemon.py 了。")
    except Exception as e:
        print(f"下载失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
