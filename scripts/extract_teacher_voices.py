# -*- coding: utf-8 -*-
"""从老师底片视频（talk.mp4）抽取「克隆干声」voice_ref.wav，并用 FunASR 转出参考文本。

背景：同科多老师（如 math / math_strict）里，**第二类（变体）老师**建档案时没有干声，
``voice_engine`` 只能是 ``edge``（edge-tts 自带音色），听起来很机械。本脚本从每位变体
老师自己的 ``talk.mp4`` 里抽出一段干净语音，配上 ASR 文本，让它们也能走 CosyVoice 零样本克隆
零样本克隆，用**素材视频里本人的声音**说话。

用法：
    python scripts/extract_teacher_voices.py            # 抽取缺失的干声
    python scripts/extract_teacher_voices.py --asr      # 抽取 + 调云端 FunASR 转写文本
    python scripts/extract_teacher_voices.py --force    # 已存在的也重抽
    python scripts/extract_teacher_voices.py --only math_strict,physics_lab

ASR 说明：CosyVoice 零样本克隆（2/3 通用）要求 ``ref_text`` 与 ``ref_audio`` **逐字对应**，
文本错了音色会明显变差，所以必须走 ASR 而不是手填。默认调算力云上的 FunASR
（127.0.0.1:10095/recognize），连接信息同 scripts/sync_cloud.py 的环境变量。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEACHERS_DIR = ROOT / "data" / "teachers"

# 干声目标：CosyVoice 零样本克隆推荐 5~15s 干净单人说
TARGET_MIN_SEC = 6.0
TARGET_MAX_SEC = 12.0
SAMPLE_RATE = 16000
# 语音占比下限：低于此值说明素材里几乎没说话（如历史底片），克隆不可靠，保持 edge
MIN_SPEECH_RATIO = 0.25

# 云端 FunASR（与 sync_cloud.py 同一套连接信息：CLOUD_* -> TUNNEL_SSH_* 回退）
CLOUD_HOST = os.getenv("CLOUD_HOST") or os.getenv("TUNNEL_SSH_HOST", "")
CLOUD_PORT = int(os.getenv("CLOUD_PORT") or os.getenv("TUNNEL_SSH_PORT") or "22")
CLOUD_USER = os.getenv("CLOUD_USER") or os.getenv("TUNNEL_SSH_USER", "root")
CLOUD_PWD = os.getenv("CLOUD_PWD") or os.getenv("TUNNEL_SSH_PASSWORD", "")
CLOUD_ROOT = os.getenv("CLOUD_ROOT", "/root/autodl-tmp/education")


def ffmpeg_exe() -> str:
    """拿到 ffmpeg 可执行文件（优先 imageio-ffmpeg 自带，其次系统 PATH）。"""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:  # noqa: BLE001
        pass
    return "ffmpeg"


def extract_audio(ffmpeg: str, video: Path, out_wav: Path) -> bool:
    """视频 -> 16k 单声道 wav。"""
    cmd = [ffmpeg, "-y", "-i", str(video), "-vn",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(out_wav)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=180)
        return out_wav.is_file() and out_wav.stat().st_size > 0
    except Exception as err:  # noqa: BLE001
        print(f"    [FAIL] ffmpeg: {err}")
        return False


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def write_wav(path: Path, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def speech_runs(data: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """能量 VAD：返回 [(起, 止)]（采样点）与语音占比。"""
    frame = int(SAMPLE_RATE * 0.02)
    hop = int(SAMPLE_RATE * 0.01)
    if len(data) < frame:
        return [], 0.0
    rms = np.array([
        float(np.sqrt(np.mean(data[i:i + frame] ** 2)))
        for i in range(0, len(data) - frame, hop)
    ])
    if rms.size == 0:
        return [], 0.0
    peak = float(np.percentile(rms, 95))
    if peak <= 0:
        return [], 0.0
    thr = max(peak * 0.10, float(np.percentile(rms, 60)) * 1.5)
    is_speech = rms > thr

    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(is_speech):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start * hop, i * hop))
            start = None
    if start is not None:
        runs.append((start * hop, len(is_speech) * hop))
    ratio = float(np.mean(is_speech))
    return runs, ratio


def pick_segment(data: np.ndarray) -> np.ndarray | None:
    """挑一段连续、够长的干净语音。"""
    runs, _ = speech_runs(data)
    if not runs:
        return None
    # 先把过短的碎段扔掉，再把相邻间隔 <0.4s 的合并（同一句话里的自然停顿）
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if end - start < SAMPLE_RATE * 0.3:
            continue
        if merged and start - merged[-1][1] < SAMPLE_RATE * 0.4:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if not merged:
        return None
    # 取最长的一段，截到目标时长
    start, end = max(merged, key=lambda r: r[1] - r[0])
    max_len = int(TARGET_MAX_SEC * SAMPLE_RATE)
    if end - start < int(TARGET_MIN_SEC * SAMPLE_RATE):
        # 最长段不够长：把后面几段拼起来补够（同一素材、同一人，可接受）
        acc_start = start
        acc_end = end
        for s, e in merged:
            if s < start:
                continue
            if acc_end - acc_start >= max_len:
                break
            acc_end = max(acc_end, e)
        start, end = acc_start, acc_end
    end = min(end, start + max_len, len(data))
    seg = data[start:end]
    # 头尾各留 80ms 静音，避免把字切掉
    pad = int(0.08 * SAMPLE_RATE)
    head = data[max(0, start - pad):start]
    tail = data[end:min(len(data), end + pad)]
    return np.concatenate([head, seg, tail]) if head.size and tail.size else seg


def asr_remote(wav: Path) -> str:
    """把 wav 传到算力云，调 FunASR /recognize 转写。"""
    import posixpath

    import paramiko

    remote_dir = posixpath.join(CLOUD_ROOT, "data", "_asr_tmp")
    remote_wav = posixpath.join(remote_dir, wav.name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(CLOUD_HOST, port=CLOUD_PORT, username=CLOUD_USER, password=CLOUD_PWD, timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            sftp.stat(remote_dir)
        except IOError:
            sftp.mkdir(remote_dir)
        sftp.put(str(wav), remote_wav)
        sftp.close()
        _, out, err = client.exec_command(
            f"curl -s -m 60 -F file=@{remote_wav} http://127.0.0.1:10095/recognize",
            timeout=120,
        )
        text = out.read().decode("utf-8", "ignore")
        if not text.strip():
            return ""
        try:
            return str(json.loads(text).get("text") or "").strip()
        except Exception:  # noqa: BLE001
            return text.strip()
    finally:
        client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", action="store_true", help="调云端 FunASR 转写参考文本")
    ap.add_argument("--force", action="store_true", help="已存在干声也重新抽取")
    ap.add_argument("--only", default="", help="只处理指定老师（逗号分隔）")
    args = ap.parse_args()

    ffmpeg = ffmpeg_exe()
    print(f"[extract] ffmpeg = {ffmpeg}")
    print(f"[extract] 教师目录 = {TEACHERS_DIR}")

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    targets = sorted(p for p in TEACHERS_DIR.iterdir() if p.is_dir())
    results: list[dict] = []

    for d in targets:
        name = d.name
        if only and name not in only:
            continue
        teacher_json = d / "teacher.json"
        video = d / "talk.mp4"
        ref = d / "voice_ref.wav"
        if not teacher_json.is_file() or not video.is_file():
            print(f"  [skip] {name}：缺 teacher.json 或 talk.mp4")
            continue
        if ref.is_file() and not args.force:
            print(f"  [skip] {name}：已有 voice_ref.wav")
            continue

        print(f"  [work] {name} ...")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "full.wav"
            if not extract_audio(ffmpeg, video, tmp):
                continue
            data = read_wav(tmp)
        _, ratio = speech_runs(data)
        if ratio < MIN_SPEECH_RATIO:
            print(f"    [SKIP] {name}：语音占比 {ratio:.2f} 过低，克隆不可靠，保持 edge")
            results.append({"teacher": name, "ok": False, "reason": f"low_speech_ratio={ratio:.2f}"})
            continue
        seg = pick_segment(data)
        if seg is None or seg.size < SAMPLE_RATE * 2:
            print(f"    [SKIP] {name}：没找到可用语音段")
            results.append({"teacher": name, "ok": False, "reason": "no_speech_segment"})
            continue

        write_wav(ref, seg)
        dur = seg.size / SAMPLE_RATE
        ref_text = ""
        if args.asr:
            try:
                ref_text = asr_remote(ref)
            except Exception as err:  # noqa: BLE001
                print(f"    [WARN] ASR 失败：{err}")
        print(f"    [OK] {name}：{dur:.1f}s  语音占比={ratio:.2f}  文本={ref_text[:40]!r}")
        results.append({
            "teacher": name, "ok": True, "seconds": round(dur, 2),
            "speech_ratio": round(ratio, 3), "ref_text": ref_text,
        })

    report = TEACHERS_DIR.parent / "raw" / "voice_extract_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 成功 {sum(1 for r in results if r['ok'])} / {len(results)}")
    print(f"[report] {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
