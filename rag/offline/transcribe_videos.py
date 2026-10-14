#!/usr/bin/env python3
"""Batch-transcribe Smart Education teaching videos with FunASR.

The decoder produces 16 kHz, mono, signed 16-bit PCM.  FSMN-VAD consumes
that PCM in small chunks and emits speech intervals.  Each interval is sent
to Paraformer, so memory usage does not grow with the length of a lesson.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


LOGGER = logging.getLogger("transcribe_videos")
SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BYTES = 2
DEFAULT_CHUNK_MS = 200
DEFAULT_VAD_MODEL = "fsmn-vad"
DEFAULT_ASR_MODEL = "paraformer-zh"
DEFAULT_PUNC_MODEL = "ct-punc"


@dataclass(frozen=True)
class VideoJob:
    path: Path
    subject: str
    grade: str
    lesson_id: str
    output: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a same-directory temporary file and replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _is_transcript(path: Path) -> bool:
    """Return whether an existing output is a complete transcript document."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("lesson_id"), str)
        and isinstance(payload.get("subject"), str)
        and isinstance(payload.get("grade"), str)
        and payload.get("source") == "smartedu"
        and isinstance(payload.get("duration_ms"), int)
        and isinstance(payload.get("segments"), list)
    )


def discover_jobs(input_root: Path, output_root: Path) -> list[VideoJob]:
    """Discover videos at exactly ``{subject}/{grade}/{lesson_id}.mp4``."""
    jobs: list[VideoJob] = []
    if not input_root.exists():
        return jobs
    candidates = (
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".mp4"
    )
    for path in sorted(candidates, key=lambda item: item.as_posix().casefold()):
        try:
            relative = path.relative_to(input_root)
        except ValueError:
            continue
        if len(relative.parts) != 3:
            LOGGER.warning("skip video outside subject/grade layout: %s", path)
            continue
        subject, grade, filename = relative.parts
        lesson_id = Path(filename).stem
        jobs.append(
            VideoJob(
                path=path,
                subject=subject,
                grade=grade,
                lesson_id=lesson_id,
                output=output_root / subject / grade / f"{lesson_id}.json",
            )
        )
    return jobs


def iter_ffmpeg_pcm(
    video_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    chunk_bytes: int = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * DEFAULT_CHUNK_MS // 1000,
) -> Iterator[bytes]:
    """Yield decoded PCM chunks from the first audio stream in ``video_path``."""
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    normal_eof = False
    try:
        while True:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            yield data
        normal_eof = True
    finally:
        if not normal_eof and process.poll() is None:
            process.terminate()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if normal_eof and return_code != 0:
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}{detail}")


def _collect_pairs(value: Any) -> list[tuple[int, int]]:
    """Find ``[start, end]`` timestamp pairs in FunASR's VAD response."""
    pairs: list[tuple[int, int]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 2 and all(
                isinstance(number, (int, float)) and not isinstance(number, bool)
                for number in item
            ):
                pairs.append((int(item[0]), int(item[1])))
                return
            for nested in item:
                visit(nested)

    visit(value)
    return pairs


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        text = result.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(result, list):
        for item in result:
            text = _result_text(item)
            if text:
                return text
    return ""


def _first_result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                return item
    return {}


def _as_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _sentence_info_segments(
    result: Any,
    *,
    interval_start_ms: int,
    interval_end_ms: int,
) -> list[dict[str, Any]]:
    """Normalize Paraformer's sentence_info to the public transcript shape."""
    payload = _first_result_dict(result)
    sentence_info = payload.get("sentence_info")
    candidates = sentence_info if isinstance(sentence_info, list) else []
    segments: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get("text"))
        if not text:
            continue
        start = _as_ms(item.get("start_ms", item.get("start")))
        end = _as_ms(item.get("end_ms", item.get("end")))
        timestamp = item.get("timestamp")
        if (start is None or end is None) and isinstance(timestamp, (list, tuple)):
            if len(timestamp) >= 2 and not isinstance(timestamp[0], (list, tuple)):
                start = _as_ms(timestamp[0])
                end = _as_ms(timestamp[1])
        if start is None or end is None:
            start, end = 0, interval_end_ms - interval_start_ms
        start = max(0, min(start, interval_end_ms - interval_start_ms))
        end = max(start, min(end, interval_end_ms - interval_start_ms))
        segments.append(
            {
                "start_ms": interval_start_ms + start,
                "end_ms": interval_start_ms + end,
                "text": text,
            }
        )
    return segments


def _recognize_interval(
    raw_path: Path,
    *,
    start_ms: int,
    end_ms: int,
    total_samples: int,
    asr_model: Any,
) -> list[dict[str, Any]]:
    """Extract one VAD interval and recognize it with timestamp-aware Paraformer."""
    start_sample = max(0, min(total_samples, math.floor(start_ms * SAMPLE_RATE / 1000)))
    end_sample = max(start_sample, min(total_samples, math.ceil(end_ms * SAMPLE_RATE / 1000)))
    if end_sample <= start_sample:
        return []

    with raw_path.open("rb") as raw:
        raw.seek(start_sample * SAMPLE_WIDTH_BYTES)
        pcm = raw.read((end_sample - start_sample) * SAMPLE_WIDTH_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_handle:
        wav_path = Path(wav_handle.name)
    try:
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm)
        result = asr_model.generate(input=str(wav_path), sentence_timestamp=True)
        segments = _sentence_info_segments(
            result,
            interval_start_ms=start_ms,
            interval_end_ms=end_ms,
        )
        if segments:
            return segments
        text = _clean_text(_result_text(result))
        if not text:
            return []
        # Some FunASR versions return text but no sentence_info. VAD bounds
        # are still precise and provide a stable, useful fallback timestamp.
        return [{"start_ms": start_ms, "end_ms": end_ms, "text": text}]
    finally:
        try:
            wav_path.unlink()
        except FileNotFoundError:
            pass


def _consume_vad_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    open_start_ms: int | None,
    emitted_end_ms: int,
) -> tuple[int | None, int, list[tuple[int, int]]]:
    emitted: list[tuple[int, int]] = []
    for start, end in pairs:
        if start >= 0 and end >= 0:
            if end > start and end > emitted_end_ms:
                emitted.append((max(start, emitted_end_ms), end))
                emitted_end_ms = end
            open_start_ms = None
        elif start >= 0 and end < 0:
            open_start_ms = start
        elif start < 0 and end >= 0:
            if open_start_ms is not None and end > open_start_ms and end > emitted_end_ms:
                emitted.append((max(open_start_ms, emitted_end_ms), end))
                emitted_end_ms = end
            open_start_ms = None
    return open_start_ms, emitted_end_ms, emitted


def transcribe_pcm_chunks(
    pcm_chunks: Iterable[bytes],
    *,
    vad_model: Any,
    asr_model: Any,
    chunk_ms: int = DEFAULT_CHUNK_MS,
) -> tuple[int, list[dict[str, Any]]]:
    """Transcribe an iterable of PCM chunks; separated for testability."""
    cache: dict[str, Any] = {}
    open_start_ms: int | None = None
    emitted_end_ms = 0
    total_samples = 0
    segments: list[dict[str, Any]] = []

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as raw_handle:
        raw_path = Path(raw_handle.name)
    try:
        iterator = iter(pcm_chunks)
        pending = next(iterator, None)
        while pending is not None:
            chunk = pending
            pending = next(iterator, None)
            if len(chunk) % SAMPLE_WIDTH_BYTES:
                raise ValueError("ffmpeg returned an odd number of PCM bytes")
            if not chunk:
                continue
            with raw_path.open("ab") as raw_file:
                raw_file.write(chunk)
            total_samples += len(chunk) // SAMPLE_WIDTH_BYTES
            waveform = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            vad_result = vad_model.generate(
                input=waveform,
                cache=cache,
                is_final=pending is None,
                chunk_size=chunk_ms,
            )
            open_start_ms, emitted_end_ms, intervals = _consume_vad_pairs(
                _collect_pairs(vad_result),
                open_start_ms=open_start_ms,
                emitted_end_ms=emitted_end_ms,
            )
            for start_ms, end_ms in intervals:
                segments.extend(
                    _recognize_interval(
                        raw_path,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        total_samples=total_samples,
                        asr_model=asr_model,
                    )
                )

        if total_samples == 0:
            raise ValueError("decoded audio is empty")

        duration_ms = math.ceil(total_samples * 1000 / SAMPLE_RATE)
        if open_start_ms is not None and duration_ms > open_start_ms:
            segments.extend(
                _recognize_interval(
                    raw_path,
                    start_ms=open_start_ms,
                    end_ms=duration_ms,
                    total_samples=total_samples,
                    asr_model=asr_model,
                )
            )
        segments.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
        return duration_ms, segments
    finally:
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass


def transcribe_video(
    video_path: Path,
    *,
    vad_model: Any,
    asr_model: Any,
    ffmpeg_bin: str = "ffmpeg",
    chunk_ms: int = DEFAULT_CHUNK_MS,
) -> tuple[int, list[dict[str, Any]]]:
    chunk_bytes = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * chunk_ms // 1000
    return transcribe_pcm_chunks(
        iter_ffmpeg_pcm(video_path, ffmpeg_bin=ffmpeg_bin, chunk_bytes=chunk_bytes),
        vad_model=vad_model,
        asr_model=asr_model,
        chunk_ms=chunk_ms,
    )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_models(
    *,
    device: str,
    vad_model_name: str = DEFAULT_VAD_MODEL,
    asr_model_name: str = DEFAULT_ASR_MODEL,
    punc_model_name: str | None = DEFAULT_PUNC_MODEL,
) -> tuple[Any, Any]:
    """Load models lazily so CLI inspection/tests work without FunASR."""
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "FunASR is not installed; run `conda activate rag_new` and "
            "`python -m pip install -r requirements.txt`"
        ) from exc

    common = {"device": device, "disable_update": True, "disable_pbar": True}
    try:
        vad_model = AutoModel(model=vad_model_name, **common)
        asr_kwargs: dict[str, Any] = {"model": asr_model_name, **common}
        if punc_model_name:
            asr_kwargs["punc_model"] = punc_model_name
        asr_model = AutoModel(**asr_kwargs)
    except TypeError:
        # Older FunASR releases do not know disable_update.
        common.pop("disable_update", None)
        vad_model = AutoModel(model=vad_model_name, **common)
        asr_kwargs = {"model": asr_model_name, **common}
        if punc_model_name:
            asr_kwargs["punc_model"] = punc_model_name
        asr_model = AutoModel(**asr_kwargs)
    return vad_model, asr_model


def load_failures(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("cannot read failure log; starting a new one: %s", path)
        return {}
    entries = payload.get("failures") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return {}
    return {
        item["input"]: item
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("input"), str)
    }


def save_failures(path: Path, failures: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(path, {"updated_at": utc_now(), "failures": list(failures.values())})


def build_transcript(
    job: VideoJob,
    *,
    duration_ms: int,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "lesson_id": job.lesson_id,
        "subject": job.subject,
        "grade": job.grade,
        "source": "smartedu",
        "duration_ms": int(duration_ms),
        "segments": [
            {
                "start_ms": int(item["start_ms"]),
                "end_ms": int(item["end_ms"]),
                "text": str(item["text"]),
            }
            for item in segments
            if str(item.get("text", "")).strip()
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("teaching_assets/raw_materials/videos"))
    parser.add_argument("--output-root", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--failed-log", type=Path, default=Path("data/logs/failed.json"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--vad-model", default=DEFAULT_VAD_MODEL)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--punc-model", default=DEFAULT_PUNC_MODEL)
    parser.add_argument("--no-punc", action="store_true", help="do not load ct-punc")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_ms <= 0:
        raise SystemExit("--chunk-ms must be positive")
    if not args.input_root.exists():
        LOGGER.error("input directory does not exist: %s", args.input_root)
        return 1

    jobs = discover_jobs(args.input_root, args.output_root)
    pending = [job for job in jobs if args.overwrite or not _is_transcript(job.output)]
    if args.max_files > 0:
        pending = pending[: args.max_files]
    LOGGER.info("discovered %d videos; %d pending", len(jobs), len(pending))
    if not pending:
        return 0

    device = resolve_device(args.device)
    LOGGER.info("loading FunASR models on %s", device)
    try:
        vad_model, asr_model = load_models(
            device=device,
            vad_model_name=args.vad_model,
            asr_model_name=args.asr_model,
            punc_model_name=None if args.no_punc else args.punc_model,
        )
    except Exception as exc:  # noqa: BLE001 - report model setup as one batch failure
        LOGGER.error("failed to load FunASR models: %s: %s", type(exc).__name__, exc)
        return 1

    failures = load_failures(args.failed_log)
    failed_this_run = 0
    for index, job in enumerate(pending, start=1):
        key = job.path.as_posix()
        LOGGER.info("[%d/%d] %s", index, len(pending), job.path)
        try:
            duration_ms, segments = transcribe_video(
                job.path,
                vad_model=vad_model,
                asr_model=asr_model,
                ffmpeg_bin=args.ffmpeg_bin,
                chunk_ms=args.chunk_ms,
            )
            atomic_write_json(
                job.output,
                build_transcript(job, duration_ms=duration_ms, segments=segments),
            )
            failures.pop(key, None)
            save_failures(args.failed_log, failures)
            LOGGER.info("wrote %s (%d segments)", job.output, len(segments))
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the batch
            failed_this_run += 1
            failures[key] = {
                "input": key,
                "output": job.output.as_posix(),
                "subject": job.subject,
                "grade": job.grade,
                "lesson_id": job.lesson_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": utc_now(),
            }
            save_failures(args.failed_log, failures)
            LOGGER.exception("failed %s", job.path)
    return 1 if failed_this_run else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
