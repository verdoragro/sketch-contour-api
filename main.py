import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Sketch Contour API")

MAX_SIDE = 1200  # processa nessa resolução e volta ao tamanho original


def _read_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Arquivo não é uma imagem válida.")
    return img


def _resize_for_processing(img: np.ndarray, max_side: int = MAX_SIDE):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_small = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        img_small = img.copy()
    return img_small, scale


def _process(img_bgr: np.ndarray, thickness: int, bg_opacity: float) -> np.ndarray:
    # 1) resize p/ processamento
    work, scale = _resize_for_processing(img_bgr, MAX_SIDE)
    H, W = work.shape[:2]

    # 2) GrabCut com retângulo amplo
    mask_gc = np.zeros((H, W), np.uint8)
    rect = (10, 10, W - 20, H - 20)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(work, mask_gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )

    # 3) Reforço por arestas próximas ao FG
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    v = np.median(gray[fg > 0]) if np.any(fg > 0) else np.median(gray)  # type: ignore
    low, high = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, low, high, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges_near_fg = cv2.bitwise_and(
        edges, edges, mask=cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=1)
    )

    # 4) Máscara final e limpeza
    mask = cv2.bitwise_or(fg, edges_near_fg)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    # 5) Contornos com buracos
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        raise ValueError("Não foram encontrados contornos.")
    area_thresh = 0.001 * H * W
    kept = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= area_thresh:
            eps = 0.003 * cv2.arcLength(cnt, True)
            kept.append(cv2.approxPolyDP(cnt, eps, True))

    # 6) Render: escurece fundo + traça contornos em preto
    # (escurecer = multiplicar pelo fator (1 - bg_opacity) nas regiões todas,
    # mantendo a imagem visível porém opaca)
    out = work.astype(np.float32)
    out *= 1.0 - bg_opacity  # escurece tudo
    out = out.astype(np.uint8)

    for cnt in kept:
        cv2.polylines(
            out, [cnt], True, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA
        )

    # 7) Redimensiona de volta ao tamanho original, se preciso
    if scale < 1.0:
        h0, w0 = img_bgr.shape[:2]
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC)

    return out


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.post("/sketch", response_class=StreamingResponse)
async def sketch(
    file: UploadFile = File(...),
    thickness: int = Query(2, ge=1, le=8, description="Espessura do traço"),
    bg_opacity: float = Query(
        0.5,
        ge=0.0,
        le=0.95,
        description="Opacidade/escurecimento do fundo (0=sem, 0.5=metade)",
    ),
):
    try:
        content = await file.read()
        img = _read_image(content)
        result = _process(img, thickness=thickness, bg_opacity=bg_opacity)

        # Codifica PNG e retorna
        ok, buf = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("Falha ao codificar PNG.")
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
