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

    # 2) Detecção de bordas mais rápida ao invés de GrabCut
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Blur leve para reduzir ruído
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny com parâmetros fixos otimizados para performance
    low, high = 50, 150
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)

    # Dilatação menor para conectar bordas próximas
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 3) Buscar contornos principais diretamente
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("Não foram encontrados contornos.")

    # Filtrar por área mínima e simplificar contornos
    area_thresh = 0.002 * H * W
    kept = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= area_thresh:
            # Simplificação mais agressiva para performance
            eps = 0.01 * cv2.arcLength(cnt, True)
            simplified = cv2.approxPolyDP(cnt, eps, True)
            if len(simplified) > 2:  # garantir que ainda é um contorno válido
                kept.append(simplified)

    # 4) Render: escurece fundo + traça contornos
    out = work.copy()

    # Escurecer o fundo
    if bg_opacity > 0:
        out = (out * (1.0 - bg_opacity)).astype(np.uint8)

    # Desenhar contornos
    for cnt in kept:
        cv2.polylines(
            out, [cnt], True, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA
        )

    # 5) Redimensiona de volta ao tamanho original, se preciso
    if scale < 1.0:
        h0, w0 = img_bgr.shape[:2]
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)

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
