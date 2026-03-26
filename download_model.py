#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预下载模型到本地缓存，并可选做一次加载验证。

用法：
    python3 download_model.py
    python3 download_model.py --model "iic/SenseVoiceSmall" --verify-load
    python3 download_model.py --model "FunAudioLLM/Fun-ASR-Nano-2512" --backend huggingface --verify-load --trust-remote-code
"""

import argparse
import sys

from model_adapter import build_auto_model


def infer_backend(model_id: str) -> str:
    if model_id.startswith("iic/") or model_id.startswith("damo/"):
        return "modelscope"
    if model_id.startswith("FunAudioLLM/"):
        return "huggingface"
    return "modelscope"


def download_from_modelscope(model_id: str) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    return snapshot_download(model_id)


def download_from_huggingface(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


def verify_with_automodel(
    model_id: str,
    device: str,
    trust_remote_code: bool,
    remote_code: str | None,
) -> str:
    if trust_remote_code or remote_code:
        print("提示：该脚本会优先使用内置模型适配逻辑，忽略手动 trust_remote_code/remote_code。")

    model = build_auto_model(model_id, device=device, disable_update=True)
    return getattr(model, "model_path", "(AutoModel 未暴露 model_path)")


def main():
    parser = argparse.ArgumentParser(description="预下载模型，并区分下载错误和加载错误")
    parser.add_argument(
        "--model",
        default="iic/SenseVoiceSmall",
        help="模型 ID，例如 iic/SenseVoiceSmall 或 FunAudioLLM/Fun-ASR-Nano-2512",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "modelscope", "huggingface"),
        default="auto",
        help="下载源，默认按模型 ID 自动推断",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="下载完成后，再用 AutoModel 做一次加载验证",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="加载验证时使用的设备，默认 cpu",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载验证时传递 trust_remote_code=True",
    )
    parser.add_argument(
        "--remote-code",
        default=None,
        help="加载验证时使用的 remote_code，例如 ./model.py",
    )
    args = parser.parse_args()

    backend = infer_backend(args.model) if args.backend == "auto" else args.backend

    print(f"准备下载模型：{args.model}")
    print(f"下载源：{backend}")
    print("步骤 1/2：仅下载到本地缓存，不做模型适配。\n")

    try:
        if backend == "modelscope":
            model_path = download_from_modelscope(args.model)
        else:
            model_path = download_from_huggingface(args.model)
    except ModuleNotFoundError as exc:
        print(f"下载失败：缺少依赖 {exc.name}", file=sys.stderr)
        print("请先安装对应下载依赖，例如：pip install modelscope huggingface_hub", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
        print("这一步失败通常说明是网络、仓库 ID、权限或下载源本身的问题。", file=sys.stderr)
        sys.exit(1)

    print(f"下载成功，本地缓存目录：{model_path}")

    if not args.verify_load:
        print("\n未执行加载验证。现在可以确认：下载链路本身是通的。")
        return

    print("\n步骤 2/2：执行 AutoModel 加载验证。\n")

    try:
        loaded_path = verify_with_automodel(
            model_id=args.model,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            remote_code=args.remote_code,
        )
    except ModuleNotFoundError as exc:
        print(f"加载验证失败：缺少依赖 {exc.name}", file=sys.stderr)
        print("这一步失败说明下载已完成，但当前运行环境不完整。", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"加载验证失败：{exc}", file=sys.stderr)
        print("这一步失败说明模型文件大概率已经下好了，但当前加载方式或参数不兼容。", file=sys.stderr)
        sys.exit(2)

    print(f"加载验证成功，模型路径：{loaded_path}")
    print("下载和基础加载都通过了。")


if __name__ == "__main__":
    main()
