"""HTTP client for 3DAwareVerification's ``perception.server`` (no import from that repo).

    from perception_gate.client import ping, verify
    ping("http://3dawareverification-dev-1:8770")          # {ok, protocol, settings}
    reply = verify(url, rgb, mask, "chair", input_elevation=30.0)
    reply["verdict"]["accepted"]

Images are numpy RGB (HxWx3 uint8) and masks HxW (nonzero = object).
"""
import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

PROTOCOL = 1


def _call(url, path, payload=None, timeout=10.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:  # 400/500 still carry {ok: false, error}
        return json.loads(error.read())


def ping(url, timeout=10.0):
    """The server's /health; raises when it is down or speaks another protocol."""
    reply = _call(url, "/health", timeout=timeout)
    if reply.get("protocol") != PROTOCOL:
        raise RuntimeError(f"perception server at {url}: protocol {reply.get('protocol')} != {PROTOCOL}")
    if not reply.get("ok"):
        raise RuntimeError(f"perception server at {url}: model worker died {reply.get('workers')}")
    return reply


def encode_png(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(data.tobytes()).decode("ascii")


def decode_png(text):
    """RGB array from a base64 image."""
    data = np.frombuffer(base64.b64decode(text), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def verify(url, rgb, mask, target, input_elevation=0.0, vocabulary=None, request_id=None, timeout=120.0):
    """POST /verify; returns {ok, verdict, view_image}. Raises on transport or server errors."""
    payload = {"protocol": PROTOCOL, "request_id": request_id, "target": target,
               "input_elevation": float(input_elevation), "vocabulary": vocabulary,
               "image": encode_png(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)),
               "mask": encode_png(((np.asarray(mask) > 0) * 255).astype(np.uint8))}
    reply = _call(url, "/verify", payload, timeout=timeout)
    if not reply.get("ok"):
        raise RuntimeError(f"perception server: {reply.get('error')}")
    return reply


def ask(url, rgb, mask, target, question, options, request_id=None, timeout=120.0):
    """POST /ask (selector jev_sam3): JEV's distribution over ``options`` for ``question`` about the masked
    object (outlined in red). Returns {probabilities, options, latency_s, n_tokens}; raises on errors."""
    payload = {"protocol": PROTOCOL, "request_id": request_id, "target": target, "question": question,
               "options": list(options),
               "image": encode_png(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)),
               "mask": encode_png(((np.asarray(mask) > 0) * 255).astype(np.uint8))}
    reply = _call(url, "/ask", payload, timeout=timeout)
    if not reply.get("ok"):
        raise RuntimeError(f"perception server /ask: {reply.get('error')}")
    return reply["jev"]


def embed(url, rgbs, request_id=None, timeout=60.0):
    """POST /embed: L2-normalised CLIP ViT-L/14 features (np.ndarray, one row per RGB frame) for the
    view-diversity gate (``perception_gate.diversity``). Raises on transport or server errors."""
    payload = {"protocol": PROTOCOL, "request_id": request_id,
               "images": [encode_png(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)) for rgb in rgbs]}
    reply = _call(url, "/embed", payload, timeout=timeout)
    if not reply.get("ok"):
        raise RuntimeError(f"perception server /embed: {reply.get('error')}")
    return np.asarray(reply["embeddings"], dtype=np.float64)


def verify_views(url, views, target, sam3="fallback", vocabulary=None, request_id=None, timeout=120.0,
                 **variants):
    """POST /verify_views (server selector d1 or jev): one decision over several views of one object,
    nearest first. ``views``: [{rgb, mask, input_elevation, step, index, range_m, scores?, jev?}] (rgb
    HxWx3 RGB, mask HxW nonzero on the object; ``scores``: SAM3 scores and ``jev``: raw JEV
    probabilities the server returned for that view before, so it is judged only once). ``variants``: grid, grid_view, crop_small, forced2, forced2_min,
    sam3_shortcut, grid_scores, forced (jev, jev_sam3). Returns {ok, verdict, grid_image}; raises on transport or
    server errors."""
    payload = {"protocol": PROTOCOL, "request_id": request_id, "target": target, "sam3": sam3,
               "vocabulary": vocabulary, **variants, "views": []}
    for view in views:
        entry = {"image": encode_png(cv2.cvtColor(np.ascontiguousarray(view["rgb"]), cv2.COLOR_RGB2BGR)),
                 "mask": encode_png(((np.asarray(view["mask"]) > 0) * 255).astype(np.uint8)),
                 "input_elevation": float(view.get("input_elevation", 0.0)), "step": int(view["step"]),
                 "index": int(view["index"]), "range_m": float(view["range_m"])}
        if view.get("scores") is not None:
            entry["scores"] = view["scores"]
        if view.get("jev") is not None:
            entry["jev"] = view["jev"]
        payload["views"].append(entry)
    reply = _call(url, "/verify_views", payload, timeout=timeout)
    if not reply.get("ok"):
        raise RuntimeError(f"perception server: {reply.get('error')}")
    return reply
