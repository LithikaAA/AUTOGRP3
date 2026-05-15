#!/usr/bin/env python3

import argparse
from pathlib import Path


def _next_token(data, index):
    n = len(data)
    while index < n:
        c = data[index]
        if c == 35:
            while index < n and data[index] not in (10, 13):
                index += 1
        elif chr(c).isspace():
            index += 1
        else:
            break

    start = index
    while index < n and not chr(data[index]).isspace():
        index += 1
    return data[start:index].decode("ascii"), index


def read_pgm(path):
    data = Path(path).read_bytes()
    magic, index = _next_token(data, 0)
    if magic not in ("P2", "P5"):
        raise ValueError(f"{path} is {magic}, expected P2 or P5 PGM")

    width, index = _next_token(data, index)
    height, index = _next_token(data, index)
    max_value, index = _next_token(data, index)
    width = int(width)
    height = int(height)
    max_value = int(max_value)
    if max_value <= 0 or max_value > 255:
        raise ValueError("Only 8-bit PGM files are supported")

    if magic == "P2":
        pixels = []
        for _ in range(width * height):
            token, index = _next_token(data, index)
            pixels.append(int(token))
        return width, height, pixels

    while index < len(data) and chr(data[index]).isspace():
        index += 1
    pixels = list(data[index:index + width * height])
    if len(pixels) != width * height:
        raise ValueError("PGM pixel data is shorter than expected")
    return width, height, pixels


def classify(pixel, occupied_threshold, unknown_threshold, unknown_as_occupied):
    # nav2 map_saver convention: occupied is dark/black, free is light/white,
    # unknown is commonly mid-gray around 205.
    if pixel <= occupied_threshold:
        return 1
    if unknown_as_occupied and pixel <= unknown_threshold:
        return 1
    return 0


def write_pbm(path, width, height, bits):
    lines = ["P1", f"{width} {height}"]
    for row in range(height):
        start = row * width
        lines.append(" ".join(str(v) for v in bits[start:start + width]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def write_csv(path, width, height, bits):
    lines = []
    for row in range(height):
        start = row * width
        lines.append(",".join(str(v) for v in bits[start:start + width]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a ROS map PGM to PBM and/or a 0/1 CSV occupancy grid."
    )
    parser.add_argument("input_pgm", help="Input .pgm map image from map_saver_cli")
    parser.add_argument("--pbm", help="Output .pbm path")
    parser.add_argument("--csv", help="Output .csv path")
    parser.add_argument(
        "--occupied-threshold",
        type=int,
        default=100,
        help="PGM values <= this are occupied/non-free. Default: 100",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=int,
        default=240,
        help="PGM values <= this are unknown/non-free when unknown is occupied. Default: 240",
    )
    parser.add_argument(
        "--unknown-as-free",
        action="store_true",
        help="Treat unknown cells as free instead of non-free.",
    )
    args = parser.parse_args()

    if not args.pbm and not args.csv:
        stem = Path(args.input_pgm).with_suffix("")
        args.pbm = str(stem) + "_binary.pbm"
        args.csv = str(stem) + "_binary.csv"

    width, height, pixels = read_pgm(args.input_pgm)
    bits = [
        classify(
            p,
            args.occupied_threshold,
            args.unknown_threshold,
            unknown_as_occupied=not args.unknown_as_free,
        )
        for p in pixels
    ]

    if args.pbm:
        write_pbm(args.pbm, width, height, bits)
        print(f"Wrote {args.pbm}")
    if args.csv:
        write_csv(args.csv, width, height, bits)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
