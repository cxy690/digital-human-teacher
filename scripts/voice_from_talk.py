# -*- coding: utf-8 -*-
"""从老师底片 talk.mp4 重建「本人音色干声」，让每位老师都用 CosyVoice3 克隆自己的声音。

为什么需要
----------
底片多为 AI 生成视频：有的音轨里只有环境声/静音（如 math_strict、physics_lab），
以前建档案时直接把**同科目主老师的干声复制**过去，于是「女形象老师」发男声
（档案 gender=female，干声却是 math 的 137Hz 男声）。

做什么（逐位老师，可 --dry-run 先看结论）
-----------------------------------------
1. ffmpeg 抽音轨 -> 16k 单声道；
2. 能量 VAD 找语音段；用 pyin 估基频，要求「有声帧够多、基频稳定」才认可是人声；
3. 可用 -> 裁 6~12s 干净干声写入 <老师>/voice_ref.wav，调云端 FunASR 转写
   ``voice_ref_text``，并把 ``voice_engine`` 置为 ``clone``；
   不可用（底片没人声）-> ``voice_engine=edge``，按 ``gender`` 给标准普通话兜底音色。

用法
----
    python scripts/voice_from_talk.py --dry-run            # 只看诊断表，不写任何文件
    python scripts/voice_from_talk.py --apply              # 重建全部
    python scripts/voice_from_talk.py --apply --only math_strict,physics_lab
    python scripts/voice_from_talk.py --apply --force      # 已有干声也重建

注意：档案已改的会先备份到 data/_voice_backup_<日期>/。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEACHERS = ROOT / "data" / "teachers"
SAMPLE_RATE = 16000
TARGET_MIN_SEC = 6.0
TARGET_MAX_SEC = 12.0

# 认可为人声的判据
# 底片是 10 秒的 AI 生成短片，老师往往只说一两句，所以下限放得很低：
# 只要 VAD 检出了像样的语音段就算「识别到音色」，克隆质量交给云端合成体检把关。
MIN_VOICED_SEC = 0.5
MIN_VOICED_FRAMES = 200     # pyin 判为有声的帧数下限（约 3.2s @ hop=256）
MAX_F0_SPREAD = 0.55        # f0 四分位距 / 中位数，越小越像人声（噪声抖动大）
F0_MIN, F0_MAX = 60.0, 400.0

MALE_VOICES = ["zh-CN-YunxiNeural", "zh-CN-YunyangNeural",
               "zh-CN-YunjianNeural", "zh-CN-YunxiaNeural"]
FEMALE_VOICES = ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"]
MALE_DEFAULT = "zh-CN-YunxiNeural"
FEMALE_DEFAULT = "zh-CN-XiaoxiaoNeural"

CLOUD_HOST = os.getenv("CLOUD_HOST") or os.getenv("TUNNEL_SSH_HOST", "")
CLOUD_PORT = int(os.getenv("CLOUD_PORT") or os.getenv("TUNNEL_SSH_PORT") or "22")
CLOUD_USER = os.getenv("CLOUD_USER", "root")
# 凭据只从环境变量读取，禁止写入代码：setx CLOUD_PWD "你的密码"
CLOUD_PWD = os.getenv("CLOUD_PWD") or os.getenv("TUNNEL_SSH_PASSWORD", "")
CLOUD_ROOT = os.getenv("CLOUD_ROOT", "/root/autodl-tmp/education")


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


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


def speech_runs(data: np.ndarray, frame_ms: float = 20.0) -> list[tuple[int, int]]:
    """能量 VAD：返回语音段（采样点区间）。"""
    frame = int(SAMPLE_RATE * frame_ms / 1000)
    hop = frame // 2
    if data.size < frame * 2:
        return []
    rms = np.array([float(np.sqrt(np.mean(data[i:i + frame] ** 2)))
                    for i in range(0, data.size - frame, hop)])
    if rms.size == 0 or float(np.max(rms)) <= 0:
        return []
    floor = float(np.percentile(rms, 20))
    peak = float(np.percentile(rms, 95))
    if peak < 0.005:                       # 整段基本静音
        return []
    # 阈值上限必须低于峰值：已有干声这种「几乎全程说话」的素材，
    # 若按 floor 抬高阈值会把大半语音判成静音（旧版就踩了这个坑）。
    thr = max(0.004, min(max(peak * 0.12, floor * 1.5), peak * 0.45))
    runs, start = [], None
    for i, val in enumerate(rms):
        if val > thr and start is None:
            start = i
        elif val <= thr and start is not None:
            end = i * hop + frame
            if end - start * hop >= SAMPLE_RATE * 0.3:
                runs.append((start * hop, end))
            start = None
    if start is not None:
        end = len(rms) * hop + frame
        if end - start * hop >= SAMPLE_RATE * 0.3:
            runs.append((start * hop, end))
    # 合并间隔 < 0.4s 的相邻段（同一句话的自然停顿）
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < SAMPLE_RATE * 0.4:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= SAMPLE_RATE * 0.5]


def voice_stats(seg: np.ndarray) -> tuple[float | None, float, int]:
    """返回 (f0 中位, f0 离散度, 有声帧数)。"""
    import librosa

    f0, voiced, _ = librosa.pyin(seg, fmin=F0_MIN, fmax=F0_MAX, sr=SAMPLE_RATE,
                                 frame_length=1024, hop_length=256)
    f0 = np.asarray(f0, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    vals = f0[voiced & np.isfinite(f0)]
    if vals.size < 50:
        return None, 9.9, int(vals.size)
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    spread = float((q3 - q1) / med) if med > 0 else 9.9
    return med, spread, int(vals.size)


def pick_segment(data: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray | None:
    """取最长的一段；不足 6s 时向后拼接，截到 12s。"""
    if not runs:
        return None
    best = max(runs, key=lambda r: r[1] - r[0])
    start, end = best
    max_len = int(TARGET_MAX_SEC * SAMPLE_RATE)
    if end - start < TARGET_MIN_SEC * SAMPLE_RATE:
        for s, e in runs:
            if s < start:
                continue
            if end - start >= max_len:
                break
            end = max(end, e)
    end = min(end, start + max_len, data.size)
    seg = data[start:end]
    pad = int(0.05 * SAMPLE_RATE)
    head = data[max(0, start - pad):start]
    tail = data[end:min(data.size, end + pad)]
    return np.concatenate([head, seg, tail])


def asr_remote(wav: Path) -> str:
    """把干声传到算力云，调 FunASR /recognize 转写（CosyVoice 要求逐字对应）。"""
    import posixpath

    import paramiko

    remote = posixpath.join(CLOUD_ROOT, "data", "_asr_tmp", wav.name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(CLOUD_HOST, port=CLOUD_PORT, username=CLOUD_USER,
                   password=CLOUD_PWD, timeout=30)
    try:
        sftp = client.open_sftp()
        try:
            sftp.stat(posixpath.dirname(remote))
        except IOError:
            sftp.mkdir(posixpath.dirname(remote))
        sftp.put(str(wav), remote)
        sftp.close()
        _, out, _err = client.exec_command(
            f"curl -s -m 120 -F file=@{remote} http://127.0.0.1:10095/recognize", timeout=180)
        text = out.read().decode("utf-8", "ignore").strip()
        if not text:
            return ""
        try:
            return str(json.loads(text).get("text") or "").strip()
        except Exception:  # noqa: BLE001
            return text
    finally:
        client.close()


def analyse(video: Path, ffmpeg: str) -> tuple[np.ndarray | None, list[tuple[int, int]], str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "full.wav"
        subprocess.run([ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(SAMPLE_RATE), "-f", "wav", str(tmp)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        if not tmp.is_file():
            return None, [], "抽取音轨失败"
        data = read_wav(tmp)
    runs = speech_runs(data)
    if not runs:
        return data, [], "底片无明显人声（静音/仅环境声）"
    voiced_sec = sum(e - s for s, e in runs) / SAMPLE_RATE
    if voiced_sec < MIN_VOICED_SEC:
        return data, runs, f"有效语音仅 {voiced_sec:.1f}s（<{MIN_VOICED_SEC}s）"
    return data, runs, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写回档案与干声（默认只诊断）")
    ap.add_argument("--force", action="store_true", help="已有干声也重建")
    ap.add_argument("--only", default="", help="只处理指定老师（逗号分隔）")
    ap.add_argument("--asr", action="store_true", default=True)
    ap.add_argument("--no-asr", dest="asr", action="store_false")
    args = ap.parse_args()

    ffmpeg = ffmpeg_exe()
    keys = [k.strip() for k in args.only.split(",") if k.strip()]
    targets = sorted(d for d in TEACHERS.iterdir() if d.is_dir() and d.name in keys) \
        if keys else sorted(d for d in TEACHERS.iterdir() if d.is_dir())
    backup = ROOT / "data" / f"_voice_backup_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"{'key':20s}{'性别':7s}{'底片人声':9s}{'有效秒':8s}{'F0中位':9s}{'离散':7s}{'结论'}")
    changed = 0
    for d in targets:
        tj, video = d / "teacher.json", d / "talk.mp4"
        if not tj.is_file() or not video.is_file():
            print(f"{d.name:20s} 缺 teacher.json / talk.mp4")
            continue
        data = json.loads(tj.read_text(encoding="utf-8"))
        gender = str(data.get("gender") or "").strip().lower()
        ref = d / "voice_ref.wav"
        # 已有干声不自动重建（--force 才重建），但**照样要看底片有没有人声**：
        # 底片读不出声音的老师必须退回 edge，不能继续拿「借来的干声」克隆。

        # 已有干声时，先体检它自己（克隆质量由这份干声决定）
        old_note = ""
        if ref.is_file():
            old = read_wav(ref)
            old_runs = speech_runs(old)
            old_sec = sum(e - s for s, e in old_runs) / SAMPLE_RATE
            om, osp, onf = voice_stats(pick_segment(old, old_runs)) if old_runs else (None, 9.9, 0)
            ok = om is not None and onf >= MIN_VOICED_FRAMES and osp <= MAX_F0_SPREAD
            heard = ("male" if om and om < 165 else "female" if om and om > 185 else "模糊") if om else "?"
            old_note = (f"现有干声 {old_sec:.1f}s F0={om and round(om)} 帧={onf} "
                        f"[{'可用' if ok else '不可用'} 听感{heard}]")

        audio, runs, why = analyse(video, ffmpeg)
        if audio is None:
            print(f"{d.name:20s}{gender:7s}-        -       -        -      {why}")
            continue
        voiced_sec = sum(e - s for s, e in runs) / SAMPLE_RATE
        seg = None
        med = spread = None
        nframes = 0
        has_talk_voice = bool(runs) and voiced_sec >= MIN_VOICED_SEC
        if has_talk_voice:
            seg = pick_segment(audio, runs)
            med, spread, nframes = voice_stats(seg) if seg is not None else (None, 9.9, 0)
            heard = ("male" if med and med < 165 else "female" if med and med > 185 else "模糊") \
                if med else "?"
            if med and heard != "模糊" and heard != gender:
                # 底片声音的性别与形象不符：克隆它就会「女形象发男声」。
                # 但如果这位老师已有一份**性别正确**的干声，就继续用那份（别降级成机械音）。
                if ref.is_file() and _gender_ok(ref, gender):
                    verdict = f"底片听感{heard} 与形象冲突，保留现有干声克隆"
                else:
                    verdict = f"性别冲突：底片听感{heard}，形象{gender}"
                    seg = None
            else:
                verdict = f"底片有人声 -> 克隆(听感{heard})"
        else:
            verdict = f"底片无人声 -> edge 兜底（{why or '静音/仅环境声'}）"

        f0s = f"{med:.0f}" if med else "-"
        sps = f"{spread:.2f}" if med else "-"
        print(f"{d.name:20s}{gender:7s}{'有' if runs else '无':9s}{voiced_sec:8.1f}{f0s:9s}{sps:7s}{verdict}   {old_note}")

        if not args.apply:
            continue

        keep_old = ref.is_file() and not args.force and has_talk_voice and _gender_ok(ref, gender)
        if seg is None and not keep_old:
            # 底片读不出本人声音：按形象性别退回 edge 标准普通话，
            # 并清掉「借来的别人的干声」，避免以后被自动脚本又改回克隆。
            _backup(tj, ref, backup, d.name)
            data["voice_engine"] = "edge"
            data["voice"] = MALE_DEFAULT if gender == "male" else FEMALE_DEFAULT
            data["voice_ref_audio"] = ""
            _keep_voice_by_gender(data)
            tj.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed += 1
            print(f"    -> edge  {data['voice']}")
            continue

        _backup(tj, ref, backup, d.name)
        if keep_old:
            # 已有干声且底片确实有人声：保留原干声，只把引擎/路径登记正确
            data["voice_ref_audio"] = f"data/teachers/{d.name}/voice_ref.wav"
            data["voice_engine"] = "clone"
            _keep_voice_by_gender(data)
            tj.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed += 1
            print(f"    -> clone  保留现有干声 ({ref.stat().st_size / 1024:.0f}KB)")
            continue

        write_wav(ref, seg)
        data["voice_ref_audio"] = f"data/teachers/{d.name}/voice_ref.wav"
        data["voice_engine"] = "clone"
        data["voice_speech_ratio"] = round(voiced_sec / max(1e-6, audio.size / SAMPLE_RATE), 3)
        if args.asr:
            try:
                text = asr_remote(ref)
            except Exception as err:  # noqa: BLE001
                print(f"    [WARN] ASR 失败：{err}")
                text = ""
            if text:
                data["voice_ref_text"] = text
            else:
                # ASR 都转不出文字，说明这段不是人在说话（音乐/噪声），别拿去克隆
                data["voice_engine"] = "edge"
                data["voice"] = MALE_DEFAULT if gender == "male" else FEMALE_DEFAULT
                data["voice_ref_audio"] = ""
                print("    [!] ASR 无文本，判定非人声 -> 改 edge")
        _keep_voice_by_gender(data)
        tj.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed += 1
        print(f"    -> clone  {seg.size / SAMPLE_RATE:.1f}s  ref_text={str(data.get('voice_ref_text'))[:40]!r}")

    if args.apply:
        print(f"\n共更新 {changed} 份档案；备份目录：{backup}")
        print("下一步：sync 到云端 -> 云端跑 calibrate_clone_voices.py 体检 -> restart_app")
    return 0


def _gender_ok(ref: Path, gender: str) -> bool:
    """现有干声的发声性别是否与形象性别一致（样本太少无法判断时视为一致）。"""
    if not ref.is_file():
        return False
    wav = read_wav(ref)
    runs = speech_runs(wav)
    if not runs:
        return False
    med, _spread, nframes = voice_stats(pick_segment(wav, runs))
    if not med or nframes < 60:
        return True                      # 判不出来就按档案走，交给云端体检
    heard = "male" if med < 165 else ("female" if med > 185 else "模糊")
    return heard == "模糊" or heard == gender


def _keep_voice_by_gender(data: dict) -> None:
    """干声性别即实际发声性别，兜底音色与之一致（clone 时只影响降级场景）。"""
    gender = str(data.get("gender") or "").strip().lower()
    voice = str(data.get("voice") or "")
    allowed = MALE_VOICES if gender == "male" else FEMALE_VOICES
    if voice not in allowed:
        data["voice"] = MALE_DEFAULT if gender == "male" else FEMALE_DEFAULT


def _backup(tj: Path, ref: Path, backup: Path, key: str) -> None:
    backup.mkdir(parents=True, exist_ok=True)
    for src, tag in ((tj, "teacher.json"), (ref, "voice_ref.wav")):
        if src.is_file():
            shutil.copy2(src, backup / f"{key}__{tag}")


if __name__ == "__main__":
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
