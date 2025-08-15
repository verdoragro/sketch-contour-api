import io

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Sketch Contour API")

# CORS bem liberal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIDE = 800  # reduzido para processar mais rápido


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

    # 2) Segmentação robusta com GrabCut + reforço por bordas
    mask_gc = np.zeros((H, W), np.uint8)
    rect = (10, 10, W - 20, H - 20)  # retângulo generoso
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut em 3 iterações para performance
    cv2.grabCut(work, mask_gc, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where(
        (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # 3) Reforço com detecção de bordas para capturar regiões escuras
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Parâmetros adaptativos baseados no histograma da região de interesse
    if np.any(fg_mask > 0):
        v = np.median(gray[fg_mask > 0])  # type: ignore
    else:
        v = np.median(gray)  # type: ignore

    low = int(max(0, 0.66 * v))
    high = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, low, high, L2gradient=True)

    # Dilatação controlada apenas próximo ao FG
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    fg_dilated = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=1)
    edges_near_fg = cv2.bitwise_and(edges_dilated, edges_dilated, mask=fg_dilated)

    # Combinar máscaras
    combined_mask = cv2.bitwise_or(fg_mask, edges_near_fg)

    # Limpeza morfológica refinada
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )

    # 4) Extração de contornos incluindo buracos internos
    contours, hierarchy = cv2.findContours(
        combined_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("Não foram encontrados contornos.")

    # Filtrar por área mínima e suavizar
    area_thresh = 0.001 * H * W
    kept = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= area_thresh:
            # Suavização mais refinada com epsilon menor
            eps = 0.002 * cv2.arcLength(cnt, True)
            simplified = cv2.approxPolyDP(cnt, eps, True)
            if len(simplified) > 2:
                kept.append(simplified)

    if len(kept) == 0:
        raise ValueError("Nenhum contorno válido após filtragem.")

    # 5) Render: escurece fundo + traça contornos suaves
    out = work.copy()

    # Escurecer o fundo
    if bg_opacity > 0:
        out = (out * (1.0 - bg_opacity)).astype(np.uint8)

    # Desenhar todos os contornos (externos e internos) com anti-aliasing
    for cnt in kept:
        cv2.polylines(
            out, [cnt], True, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA
        )

    # 6) Redimensiona de volta ao tamanho original
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
