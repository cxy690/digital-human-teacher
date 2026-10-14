"""数字人服务启动入口。

用法（在 digital_human 目录下）：
    python run.py                          # 默认 8010 端口、ai3 形象
    python run.py --port 8010 --avatar ai3 --batch-size 8
"""
import argparse
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 保证 musetalk / server 包优先解析到本项目目录（不依赖启动时的工作目录）
sys.path.insert(0, BASE_DIR)

from aiohttp import web  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="数字人实时服务（MuseTalk 推理核心）")
    parser.add_argument("--host", default=None, help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认 8010）")
    parser.add_argument("--avatar", default=None, help="数字人形象 ID（默认 ai3）")
    parser.add_argument("--avatar-root", default=None, help="素材根目录（默认项目级 data/avatars）")
    parser.add_argument("--batch-size", type=int, default=None, help="推理批大小（默认 8）")
    parser.add_argument("--profile", default=None,
                        help="配置档：cloud→加载 config.cloud.yaml（4090/24GB 放开节流）；"
                             "默认 config.yaml（本机 8GB）。等价于环境变量 DH_PROFILE")
    parser.add_argument("--config", default=None, help="直接指定配置文件路径（优先于 --profile 与默认）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from server.app import create_app
    from server.config import load_config
    from server.engine import DigitalHumanEngine

    if args.profile:
        os.environ["DH_PROFILE"] = args.profile

    cfg = load_config(path=args.config, overrides={
        "listen_host": args.host,
        "listen_port": args.port,
        "avatar_id": args.avatar,
        "avatar_root": args.avatar_root,
        "batch_size": args.batch_size,
    })

    engine = DigitalHumanEngine(cfg)
    app = create_app(engine, cfg)
    logging.getLogger("digital_human").info(
        "服务启动：http://%s:%s（avatar=%s）", cfg["listen_host"], cfg["listen_port"], cfg["avatar_id"])
    web.run_app(app, host=cfg["listen_host"], port=cfg["listen_port"], print=None)


if __name__ == "__main__":
    main()
