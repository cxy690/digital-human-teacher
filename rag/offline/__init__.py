# -*- coding: utf-8 -*-
"""rag.offline —— 离线接入流水线包（原始素材 → 解析 → 清洗 → 切块 → 标注 → 出库）。

各 stage 既支持直接脚本运行（``python rag/offline/s1_parse.py``），
也支持模块方式（``python -m rag.offline.s1_parse``）。
共享底座见 :mod:`rag.offline.common`。
"""
