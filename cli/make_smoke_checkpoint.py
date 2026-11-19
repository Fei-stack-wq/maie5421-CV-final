"""Create local random weights, never download or train a detector."""

import argparse
from pathlib import Path

from cli.smoke_model import WARNING, create_smoke_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成未训练的本地流程测试权重。")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_smoke"))
    args = parser.parse_args(argv)
    try:
        checkpoint, config = create_smoke_checkpoint(args.output_dir)
    except OSError as exc:
        parser.error(str(exc))
    print(WARNING)
    print(f"checkpoint: {checkpoint}\nconfig: {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
