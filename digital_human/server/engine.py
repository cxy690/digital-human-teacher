"""数字人渲染引擎：待机帧循环 + 播报任务串行渲染。

线程模型：
- _worker 后台线程消费播报任务；空闲时以 fps 节奏循环待机帧。
- WebRTC 轨道在自己的事件循环里按各自时钟读取最新帧/音频快照（本引擎只做线程安全的快照读写）。

素材（数据包 bundle）：默认素材（avatar_id）用于待机与普通播报；echo 可带 expression
指定表情素材（素材目录名，如 ai3_2/ai3_3），首次使用时懒加载并缓存，缺失时回退默认素材。

推理流程（MuseTalk 官方推理核心的服务化封装）：
文本 → TTS(16k) → whisper 音频特征（每帧一组）→ UNet 批推理 → VAE 逐帧分块解码 → 贴回原帧（blending）
"""
import glob
import logging
import math
import os
import pickle
import queue
import threading
import time

import cv2
import numpy as np
import torch
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import datagen, load_all_model

from server.tts import EdgeTTS

logger = logging.getLogger("digital_human.engine")


class DigitalHumanEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fps = int(cfg["fps"])
        self.batch_size = int(cfg["batch_size"])
        # VAE 解码档位：本机 8GB 逐帧（避免显存分页）；云端 24GB 整批（省循环/launch 开销）
        self.vae_batch_decode = bool(cfg.get("vae_batch_decode", False))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._load_models()
        self._load_avatar()
        self.tts = EdgeTTS(**cfg["tts"])

        # ---- 自适应播报帧率（根治“长文本嘴冻住”）----
        # 播放时钟按 fps 取帧：产出速率必须 ≥ fps，否则帧缓冲被音频时钟抽干，
        # 显示端只能反复沿用上一帧（表现为嘴不动）。目标 fps 只是“期望值”，
        # 真实吞吐受 GPU/显存档位/批大小/并发负载影响，故按实测吞吐自动降档。
        self.min_fps = int(cfg.get("min_fps", 12))            # 自适应下限
        self.fps_margin = float(cfg.get("fps_margin", 0.85))  # 安全系数：fps ≤ 吞吐×该值
        self.preroll_seconds = float(cfg.get("preroll_seconds", 1.2))  # 开播预滚时长
        self._render_fps = float(self.fps)                    # 实测渲染吞吐（EWMA，扣背压）
        self._play_fps = float(self.fps)                      # 当前播报实际使用的 fps

        # ---- 输出快照（供 WebRTC 轨道读取）----
        self._frame_lock = threading.Lock()
        self._latest_frame = None            # np BGR 全帧（待机）
        self._play_t0 = None                 # 播报音频起播墙钟；None=未在播报
        self._play_frames = {}               # 口型帧序号 -> 帧（按音频时钟取用）
        self._play_chunk = 0                 # 已产出的口型帧数（下一帧序号）
        self._audio_lock = threading.Lock()
        self._audio_pcm = None               # 当前播报段 48k int16 音频（None=静音）
        self._audio_seq = 0                  # 音频段序号（轨道据此重置自己的读指针，支持多轨道各读各的）

        # ---- 任务与状态 ----
        self._tasks = queue.Queue()
        self._speaking = threading.Event()
        self._interrupt = threading.Event()
        self._stopped = threading.Event()
        self._idx = 0                        # 全局帧序号（在素材帧上连续循环）

        # 渲染吞吐初值：启动自测一次（静音跑通全链路），供首条播报的自适应 fps 使用；
        # 不做自测时以目标 fps 为初值（首条长文本仍有欠帧风险）
        if int(cfg.get("warmup_frames", 16)) > 0:
            self._benchmark()

        self._worker = threading.Thread(target=self._loop, name="dh-render", daemon=True)
        self._worker.start()
        logger.info("数字人引擎就绪（avatar=%s, device=%s, batch=%d, 目标 fps=%d, 吞吐初值 %.1f fps）",
                    cfg["avatar_id"], self.device, self.batch_size, self.fps, self._render_fps)

    # ------------------------------------------------------------------ 初始化
    def _load_models(self):
        cfg = self.cfg
        logger.info("加载 MuseTalk 模型...")
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=os.path.join(cfg["musetalk_dir"], "unet.pth"),
            vae_type="sd-vae",
            unet_config=os.path.join(cfg["musetalk_dir"], "musetalk.json"),
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path=cfg["whisper_dir"])
        self.whisper = WhisperModel.from_pretrained(cfg["whisper_dir"])
        self.whisper = self.whisper.to(device=self.device, dtype=self.unet.model.dtype).eval()
        self.whisper.requires_grad_(False)

    def _load_avatar(self):
        """加载默认素材（待机与无表情播报使用）；表情素材首次使用时懒加载。"""
        self._bundles = {}
        self._bundle_failed = set()
        self._default_id = self.cfg["avatar_id"]
        self._bundles[self._default_id] = self._load_bundle(self._default_id)

    def _load_bundle(self, avatar_id):
        """加载一个素材目录为渲染数据包（frames/masks/coords/latents）。"""
        d = os.path.join(self.cfg["avatar_root"], avatar_id)
        logger.info("加载数字人素材：%s", d)
        with open(os.path.join(d, "coords.pkl"), "rb") as f:
            coords = pickle.load(f)
        with open(os.path.join(d, "mask_coords.pkl"), "rb") as f:
            mask_coords = pickle.load(f)
        # 平滑框仅用于稳定贴回位置；原始框保留：latent/mask 均按原始框生成，
        # _compose 里把 mask 随框一起仿射变换，保证生成内容与融合掩膜严格对齐
        coords_smooth = self._smooth_coords(coords)
        latents = torch.load(os.path.join(d, "latents.pt"), map_location="cpu")
        frames = self._read_imgs(sorted(glob.glob(os.path.join(d, "full_imgs", "*.png"))))
        masks = self._read_imgs(sorted(glob.glob(os.path.join(d, "mask", "*.png"))))
        # 闭嘴兜底帧（scripts/gen_still_base.py 离线静音渲染产出）：底图取自讲话录像、
        # 任意帧嘴都是张开的，非生成期直接送底图会表现为"没人说话嘴却在动"
        still_path = os.path.join(d, "still.png")
        still = cv2.imread(still_path) if os.path.exists(still_path) else None
        if still is None:
            logger.warning("素材缺少 still.png（闭嘴兜底帧），非生成期回退底图原帧：%s", avatar_id)
        n = len(frames)
        assert n > 0 and n == len(coords) == len(masks) == len(mask_coords) == len(latents), \
            "素材数据数量不一致，请重新生成预处理数据"
        logger.info("素材加载完成：%s（%d 帧，still=%s）", avatar_id, n, still is not None)
        return {"frames": frames, "masks": masks, "coords": coords,
                "coords_smooth": coords_smooth,
                "mask_coords": mask_coords, "latents": latents, "still": still}

    @staticmethod
    def _smooth_coords(coords, window=9):
        """人脸框时序平滑（移动平均，window 取奇数）。

        逐帧 landmarks 的 bbox 上/下缘带 ±数像素高频噪声（upper_bond=2*鼻-下巴 会放大它），
        裁剪框忽大忽小 → 贴回区域逐帧跳动、嘴部尺寸闪烁；平均掉高频、保留头部慢速运动。
        """
        a = np.asarray(coords, dtype=np.float32)
        if window < 3 or len(a) < window:
            return a
        half = window // 2
        pad = np.pad(a, ((half, half), (0, 0)), mode="edge")
        csum = np.cumsum(pad, axis=0)
        csum = np.vstack([np.zeros((1, a.shape[1]), dtype=np.float32), csum])
        return (csum[window:] - csum[:-window]) / window

    def _resolve_bundle(self, expression):
        """取表情对应素材包：懒加载 + 缓存；素材缺失回退默认素材并记入负缓存。"""
        name = (expression or "").strip() or self._default_id
        if name == self._default_id or name in self._bundle_failed:
            return self._bundles[self._default_id]
        if name in self._bundles:
            return self._bundles[name]
        try:
            bundle = self._load_bundle(name)
        except Exception as err:  # noqa: BLE001 - 素材缺失不中断播报
            logger.warning("表情素材 %s 不可用（%s），本次回退默认素材", name, err)
            self._bundle_failed.add(name)
            return self._bundles[self._default_id]
        self._bundles[name] = bundle
        return bundle

    def _switch_avatar(self, avatar_id):
        """切换激活素材（待机与默认播报同源）；素材缺失时保持原素材不变。

        仅在渲染线程调用（echo/set_avatar 均投递任务，由 _loop 串行消费）。
        """
        name = (avatar_id or "").strip()
        if not name or name == self._default_id:
            return
        if name in self._bundle_failed:
            return
        if name not in self._bundles:
            try:
                self._bundles[name] = self._load_bundle(name)
            except Exception as err:  # noqa: BLE001 - 素材缺失不中断现有播报
                logger.warning("切换素材 %s 失败（%s），沿用 %s", name, err, self._default_id)
                self._bundle_failed.add(name)
                return
        self._default_id = name
        self._idx = 0
        still = self._bundles[name].get("still")
        if still is not None:                      # 立即换待机帧，避免切换瞬间闪回旧素材
            with self._frame_lock:
                self._latest_frame = still
        logger.info("激活素材切换为：%s", name)

    @staticmethod
    def _read_imgs(paths):
        return [cv2.imread(p) for p in paths]

    # ------------------------------------------------------------------ 对外接口
    def echo(self, text, voice=None, expression=None, avatar=None):
        """入队一条播报任务。

        expression：表情素材目录名（如 ai3_2；None=当前激活素材）；
        avatar：本次播报要切换到的「激活素材」（素材目录名，如 ai4）。切换后待机动画与
        后续播报同源，避免"讲到一半换了个人"；None=沿用当前激活素材。
        """
        text = (text or "").strip()
        if not text:
            return
        self._tasks.put((text, voice, expression, avatar))

    def set_avatar(self, avatar_id):
        """请求切换「激活素材」（待机与播报同源）。

        线程安全：只把切换意图投递到渲染线程执行（不在此直接改引擎状态），
        避免与待机循环 _idle_tick / 播报取素材产生竞态。
        """
        name = (avatar_id or "").strip()
        if not name:
            return
        self._tasks.put(("", None, None, name))

    def active_avatar(self):
        """当前激活素材目录名（待机与默认播报使用）。"""
        return self._default_id

    def is_speaking(self):
        """是否有播报任务在执行（含 TTS 合成、特征提取、开播预滚等准备阶段）。"""
        return self._speaking.is_set()

    def is_playing(self):
        """口型数据是否已就绪并开始播放（音频/画面真正在播）。

        与 is_speaking 的区别：准备阶段（TTS 合成、whisper 特征、攒预滚帧）is_speaking
        已为真而 is_playing 仍为假。显示端据此在准备期保持待机动画，等真正出帧再切实时层；
        否则会过早切到尚无新帧的实时画面 —— 人物停在上一帧，看起来像卡死。
        """
        with self._frame_lock:
            return self._play_t0 is not None

    def interrupt(self):
        """打断当前播报并清空等待队列。"""
        while True:
            try:
                self._tasks.get_nowait()
            except queue.Empty:
                break
        self._interrupt.set()

    def current_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def frame_at(self, now):
        """按音频时钟取口型帧：返回 (帧|None, 是否播报中)。

        显示端（WebRTC 轨道）固定节拍调用：帧 j 应在 t0+j/fps 展示。
        批推理尚未产出该帧时返回 None，由显示端沿用上一帧——
        旧实现整批瞬间发布、显示端只读到每批末尾帧，中间口型全被跳过（嘴部大跳/抖动）。
        """
        with self._frame_lock:
            if self._play_t0 is None:
                return self._latest_frame, False
            j = max(int((now - self._play_t0) * self._play_fps), 0)
            frame = self._play_frames.get(j)
            if frame is None:
                return None, True
            return frame, True

    def audio_snapshot(self):
        """返回 (pcm48_int16 | None, 段序号)。轨道按段序号变化重置自己的读指针。"""
        with self._audio_lock:
            return self._audio_pcm, self._audio_seq

    # ------------------------------------------------------------------ 主循环
    def _loop(self):
        interval = 1.0 / self.fps
        while not self._stopped.is_set():
            try:
                text, voice, expression, avatar = self._tasks.get(timeout=interval)
            except queue.Empty:
                self._idle_tick()
                continue
            try:
                self._switch_avatar(avatar)   # 先切素材：待机帧与本次播报同源
                if not text:                  # 纯切换请求（POST /api/avatar）
                    continue
                self._speak(text, voice, expression)
            except Exception:
                logger.exception("播报失败：%s", text[:30])
            finally:
                self._speaking.clear()
                self._interrupt.clear()
                with self._audio_lock:
                    self._audio_pcm = None
                    self._audio_seq += 1
                with self._frame_lock:
                    self._play_t0 = None
                    self._play_frames.clear()

    def _idle_tick(self):
        """待机：输出闭嘴兜底帧（缺 still 时回退底图帧循环）。"""
        bundle = self._bundles[self._default_id]
        frame = bundle.get("still")
        if frame is None:
            frame = bundle["frames"][self._idx % len(bundle["frames"])]
        with self._frame_lock:
            self._latest_frame = frame
        self._idx += 1

    def _speak(self, text, voice, expression=None):
        self._interrupt.clear()
        self._speaking.set()
        bundle = self._resolve_bundle(expression)
        # 本条播报的自适应 fps：以实测渲染吞吐为上限，保证产出 ≥ 播放时钟（否则欠帧冻嘴）
        task_fps = self._pick_play_fps()
        buffer_max = max(48, int(task_fps * 6))            # 时间轴缓冲上限（6s）
        preroll = max(1, int(task_fps * self.preroll_seconds))   # 开播预滚帧数
        # 开播前（TTS/推理窗口）显示端会读 latest 帧：先换成闭嘴兜底帧，
        # 避免预滚期露出上一段播报的张嘴帧或底图原帧
        with self._frame_lock:
            if bundle.get("still") is not None:
                self._latest_frame = bundle["still"]
            self._play_frames.clear()
            self._play_chunk = 0
            self._play_t0 = None
        base = self._idx
        t0 = time.time()
        logger.info("开始播报：%s（素材：%s，fps=%.0f）", text[:30], expression or self._default_id, task_fps)

        # 1) TTS
        y16, pcm48 = self.tts.synthesize(text, voice)
        if len(y16) == 0:
            logger.warning("TTS 返回空音频，跳过本条")
            return
        logger.info("TTS 完成：语音 %.2fs，耗时 %.2fs", len(y16) / 16000.0, time.time() - t0)

        # 2) 音频特征（每帧一组 whisper 特征；窗口 padding 随本条 fps 折算）
        chunks = self._whisper_chunks(y16, task_fps)

        # 3) 批推理 → 逐帧贴回原帧
        gen = datagen(chunks, bundle["latents"], batch_size=self.batch_size, delay_frame=base)
        audio_start_time = None
        prod_t0 = time.time()          # 生产吞吐测量起点
        wait_total = 0.0               # 背压等待累计（测原始吞吐时扣除，避免被节流掩盖真实产能）
        batch_no = 0
        last_prod = None
        compose_ms = 0.0
        for whisper_batch, latent_batch in gen:
            if self._interrupt.is_set():
                break
            res_frames = self._infer(whisper_batch, latent_batch)
            for res in res_frames:
                if self._interrupt.is_set():
                    break
                # 背压：开播后缓冲积压达上限时等播放头推进（开播前不限制，尽快攒够预滚）。
                # 缓冲内保留全部“未播”帧（一帧不丢），只清理播放头之前的旧帧；
                # 播放头 j_now 按墙钟前进，无浏览器连接时也会推进放行，不会卡死播报。
                w0 = time.time()
                while not self._interrupt.is_set():
                    with self._frame_lock:
                        if self._play_t0 is None:
                            break
                        j_now = int((time.time() - self._play_t0) * self._play_fps)
                        for stale in [k for k in self._play_frames if k < j_now - 2]:
                            del self._play_frames[stale]
                        if len(self._play_frames) < buffer_max:
                            break
                    time.sleep(0.03)
                wait_total += time.time() - w0
                if self._interrupt.is_set():
                    break
                t_c = time.time()
                frame = self._compose(bundle, res, self._idx)
                compose_ms += (time.time() - t_c) * 1000.0
                with self._frame_lock:
                    # 入时间轴缓冲：帧就绪后绝不丢弃，由 frame_at 按音频时钟逐帧取用
                    self._play_frames[self._play_chunk] = frame
                    self._play_chunk += 1
                    if self._play_t0 is not None:
                        self._latest_frame = frame
                self._idx += 1
                last_prod = time.time()
                # 预滚达标才开播：先攒够 preroll 帧再出声，吸收“批次间推理空档”，
                # 避免显示端在两次批次之间取不到帧而反复沿用上帧（长文本尤其明显）
                if audio_start_time is None and self._play_chunk >= preroll:
                    audio_start_time = self._start_playback(pcm48, task_fps)
            if audio_start_time is not None:
                logger.info("批 %d 口型帧就绪：+%.2fs（按 %.0ffps 应 +%.2fs，合成 %.0fms/帧）",
                            batch_no, time.time() - audio_start_time, task_fps,
                            self._play_chunk / task_fps, compose_ms / max(1, len(res_frames)))
            else:
                logger.info("批 %d 已缓存 %d/%d 帧（预滚中）", batch_no, self._play_chunk, preroll)
            batch_no += 1
            compose_ms = 0.0

        # 整条短于预滚：生成结束仍未开播 → 此刻开播（帧已全部就绪，不可能欠帧）
        if audio_start_time is None and self._play_chunk > 0 and not self._interrupt.is_set():
            audio_start_time = self._start_playback(pcm48, task_fps)

        # 4) 更新渲染吞吐估计（扣除背压等待 → 真实产能），供下一条播报选 fps
        if last_prod is not None and self._play_chunk > 0:
            span = (last_prod - prod_t0) - wait_total
            if span > 0.2:
                rate = self._play_chunk / span
                self._render_fps = 0.6 * self._render_fps + 0.4 * rate
                logger.info("渲染吞吐：%.1f fps（EWMA %.1f，扣除背压 %.1fs）→ 下一条自适应 fps=%.0f",
                            rate, self._render_fps, wait_total, self._pick_play_fps())
        if audio_start_time is not None and last_prod is not None:
            lead = self._play_chunk / task_fps - (last_prod - audio_start_time)
            logger.info("播报产出：%d 帧 / 音频 %.2fs（生成结束时领先播放头 %.2fs）",
                        self._play_chunk, len(pcm48) / 48000.0, lead)

        # 5) 等音频播完再回待机（串行处理后续任务，避免声音重叠）
        if audio_start_time is not None and not self._interrupt.is_set():
            audio_dur = len(pcm48) / 48000.0
            while not self._interrupt.is_set():
                if time.time() - audio_start_time >= audio_dur:
                    break
                time.sleep(0.02)

    # ---------------------------------------------------------------- 自适应帧率
    def _pick_play_fps(self):
        """按实测渲染吞吐选定本条播报的 fps。

        约束：产出速率必须 ≥ 播放时钟 fps，否则帧缓冲被音频时钟抽干 → 显示端反复
        沿用上帧（嘴冻住）。目标 fps 只是期望值，故取
        ``min(目标 fps, 实测吞吐 × 安全系数)`` 并夹在 [min_fps, 目标 fps] 内。
        """
        target = int(self.fps)
        cap = int(math.floor(self._render_fps * self.fps_margin))
        fps = min(target, max(self.min_fps, min(target, cap)))
        return float(max(1, fps))

    def _start_playback(self, pcm48, fps):
        """开播：发布整段音频、并把播放时钟/帧序号归零对齐（此刻缓冲内已有预滚帧）。"""
        now = time.time()
        with self._audio_lock:
            self._audio_pcm = pcm48
            self._audio_seq += 1
        with self._frame_lock:
            self._play_fps = fps
            self._play_t0 = now
        logger.info("开始出声（预滚 %.2fs，fps=%.0f）", self._play_chunk / fps, fps)
        return now

    def _infer(self, whisper_batch, latent_batch):
        """一批音频特征 + 素材 latent → 解码后的口型帧（numpy HWC 列表）。"""
        audio_batch = self.pe(whisper_batch.to(self.device))
        latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
        with torch.no_grad():
            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_batch).sample
        # VAE 解码策略按显存档位切换：
        # - 本机 8GB（vae_batch_decode=False）：逐帧 decode 让 GPU 连转、每批仅同步/回传一次；
        #   整批 decode 会触发 Windows 显存分页（实测 937ms/帧），必须避免。
        # - 云端 4090/24GB（vae_batch_decode=True）：显存充足，整批一次 decode 省逐帧 launch/循环开销。
        if self.vae_batch_decode:
            res_batch = self.vae.decode_latents_gpu(pred_latents).cpu().numpy()
        else:
            outs = [self.vae.decode_latents_gpu(pred_latents[k:k + 1])
                    for k in range(pred_latents.shape[0])]
            res_batch = torch.cat(outs, dim=0).cpu().numpy()
        return [res_batch[k] for k in range(res_batch.shape[0])]

    def _benchmark(self):
        """启动自测：用静音音频跑一遍「whisper 特征 → UNet → VAE → 贴回」全链路，
        实测渲染吞吐（fps），作为自适应 fps 的初值。

        必要性：目标 fps 只是期望值，真实吞吐取决于 GPU 型号 / 显存档位 / 批大小 /
        并发负载。若目标 > 真实吞吐，首条长文本就会因产出慢于播放时钟而欠帧冻嘴；
        先测一次，第一条播报即可选到可持续的 fps。
        """
        try:
            frames = max(4, int(self.cfg.get("warmup_frames", 16)))
            silence = np.zeros(int(16000 * frames / max(1, self.fps)), dtype=np.float32)
            chunks = self._whisper_chunks(silence, self.fps)
            bundle = self._bundles[self._default_id]
            gen = datagen(chunks, bundle["latents"], batch_size=self.batch_size, delay_frame=0)
            n, idx, t0 = 0, 0, time.time()
            for whisper_batch, latent_batch in gen:
                for res in self._infer(whisper_batch, latent_batch):
                    self._compose(bundle, res, idx)
                    idx += 1
                    n += 1
                    if n >= frames:
                        break
                if n >= frames:
                    break
            dt = time.time() - t0
            if n >= 4 and dt > 0.05:
                self._render_fps = n / dt
                logger.info("启动自测：渲染吞吐 %.1f fps（目标 %d）→ 自适应 fps 初值 %.0f",
                            self._render_fps, self.fps, self._pick_play_fps())
            else:
                logger.warning("启动自测样本不足（%d 帧 / %.2fs），沿用目标 fps=%d", n, dt, self.fps)
        except Exception as err:  # noqa: BLE001 - 自测失败不影响服务启动
            logger.warning("启动自测失败（%s），沿用目标 fps=%d", err, self.fps)

    def _whisper_chunks(self, y16, fps=None):
        """16k 语音 → 每帧一组 whisper 特征（官方 AudioProcessor.get_audio_feature 的内存版）。

        fps 为本条播报实际使用的帧率（自适应后可能低于目标 fps）：帧数、特征窗 padding
        都必须与播放时钟同源，否则口型与音素错位。
        """
        fps = int(fps or self.fps)
        sr = 16000
        seg_len = 30 * sr
        segments = [y16[i:i + seg_len] for i in range(0, len(y16), seg_len)]
        feats = []
        for seg in segments:
            feat = self.audio_processor.feature_extractor(
                seg, return_tensors="pt", sampling_rate=sr).input_features
            feats.append(feat)
        # 特征窗按帧数加 padding：官方基准 25fps±2 帧（±80ms），换算成时长约 0.067s，
        # 按当前 fps 折算帧数保证窗口时长一致——18fps→±1 帧(±56ms)，30fps→±2 帧(±67ms)；
        # 窗口过宽口型模糊/与音素错位，过窄则失去上下文，均破坏音画同步
        pad = max(1, round(fps * 2 / 25))
        return self.audio_processor.get_whisper_chunk(
            feats, self.device, self.unet.model.dtype, self.whisper,
            len(y16), fps=fps,
            audio_padding_length_left=pad, audio_padding_length_right=pad)

    def _compose(self, bundle, res_frame, idx):
        """口型帧（256x256）贴回素材帧。

        贴回位置用平滑框 S（消逐帧检测抖动）；生成内容与 mask 均产自原始框 B，
        故把 mask 随 B→S 仿射一起变换、融合窗同步平移缩放——生成内容与掩膜
        严格对齐（错贴会导致下半脸扭曲），同时贴回位置跨帧稳定。
        """
        frames = bundle["frames"]
        i = idx % len(frames)
        ori = frames[i]
        x1, y1, x2, y2 = [float(v) for v in bundle["coords"][i]]
        sx1, sy1, sx2, sy2 = [float(v) for v in bundle["coords_smooth"][i]]
        kx = (sx2 - sx1) / max(1.0, x2 - x1)
        ky = (sy2 - sy1) / max(1.0, y2 - y1)
        tx = sx1 - kx * x1
        ty = sy1 - ky * y1
        fw, fh = int(round(sx2 - sx1)), int(round(sy2 - sy1))
        face = cv2.resize(res_frame.astype(np.uint8), (fw, fh)) #缩放生成口型帧

        cx1, cy1, cx2, cy2 = [float(v) for v in bundle["mask_coords"][i]]
        # 融合窗 = 变换后的 mask 窗与平滑贴回框的并集（保证贴回偏移非负）
        nx1 = int(min(kx * cx1 + tx, sx1)) // 2 * 2
        ny1 = int(min(ky * cy1 + ty, sy1)) // 2 * 2
        nx2 = (int(max(kx * cx2 + tx, sx2)) + 3) // 2 * 2   # 留 ≥2px 余量防贴回框越界
        ny2 = (int(max(ky * cy2 + ty, sy2)) + 3) // 2 * 2
        # mask 存的是以 mask_coords 为原点的裁剪小窗（非全帧）：仿射平移量必须
        # 加上小窗原点（kx*cx1），否则采样全落在小窗外 → 掩膜恒全零 → 生成的嘴贴不上去
        m = np.float32([[kx, 0, kx * cx1 + tx - nx1], [0, ky, ky * cy1 + ty - ny1]])
        mask = cv2.warpAffine(bundle["masks"][i], m, (nx2 - nx1, ny2 - ny1),
                              flags=cv2.INTER_LINEAR, borderValue=0)
        return get_image_blending(ori, face, (nx1 + int(round(sx1 - nx1)), ny1 + int(round(sy1 - ny1)),
                                              nx1 + int(round(sx1 - nx1)) + fw,
                                              ny1 + int(round(sy1 - ny1)) + fh),
                                  mask, (nx1, ny1, nx2, ny2))
