"""Image <-> numpy conversion without cv_bridge (avoids NumPy ABI problems)."""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, np.uint8)
    if msg.encoding == "bgr8":
        return buf.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        return buf.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if msg.encoding in ("bgra8", "rgba8"):
        a = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
        return a if msg.encoding == "bgra8" else a[:, :, ::-1].copy()
    raise ValueError(f"unsupported encoding {msg.encoding}")


def bgr_to_imgmsg(img: np.ndarray, header) -> Image:
    m = Image()
    m.header, m.height, m.width = header, img.shape[0], img.shape[1]
    m.encoding, m.step = "bgr8", img.shape[1] * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


def imgmsg_to_f32(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)


def f32_to_imgmsg(arr: np.ndarray, header) -> Image:
    arr = np.ascontiguousarray(arr, np.float32)
    m = Image()
    m.header, m.height, m.width = header, arr.shape[0], arr.shape[1]
    m.encoding, m.step = "32FC1", arr.shape[1] * 4
    m.data = arr.tobytes()
    return m
