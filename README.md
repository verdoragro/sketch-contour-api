# Sketch Contour API

API FastAPI para conversão de imagens em esboços com contornos destacados.

## Funcionalidades

- **Processamento de imagens**: Converte fotos em esboços com contornos bem definidos
- **Segmentação inteligente**: Usa GrabCut para separar objeto principal do fundo
- **Detecção de bordas**: Aplica Canny para realçar contornos importantes
- **Parâmetros ajustáveis**: Controle de espessura do traço e opacidade do fundo

## Endpoints

### `GET /health`
Verifica se a API está funcionando.

**Resposta:**
```json
{"status": "ok"}
```

### `POST /sketch`
Processa uma imagem e retorna um esboço com contornos.

**Parâmetros:**
- `file`: Arquivo de imagem (FormData)
- `thickness`: Espessura do traço (1-8, padrão: 2)
- `bg_opacity`: Opacidade do fundo (0.0-0.95, padrão: 0.5)

**Retorna:** Imagem PNG processada

## Como funciona

1. **Redimensionamento**: Processa em resolução máxima de 1200px para otimização
2. **Segmentação**: GrabCut separa objeto principal do fundo
3. **Detecção de bordas**: Canny identifica contornos importantes
4. **Limpeza**: Operações morfológicas refinam a máscara
5. **Renderização**: Escurece fundo e desenha contornos em preto
6. **Retorno**: Redimensiona para tamanho original

## Tecnologias

- FastAPI
- OpenCV
- NumPy