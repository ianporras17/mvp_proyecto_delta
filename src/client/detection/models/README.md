# Recurso LPD-YuNet

Archivo FP32: `license_plate_detection_lpd_yunet_2023mar.onnx`.

- Fuente oficial: https://huggingface.co/opencv/license_plate_detection_yunet
- Revisión: `962e8d8849878106cb84039dab61cf97bb683b31`.
- SHA-256: `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7`.
- Licencia del recurso: Apache 2.0, reproducida en `LICENSE`.

El modelo se carga localmente, no se descarga durante la inferencia. El tamaño
nativo de la imagen BGR se conserva; `blobFromImage(image)` sigue el ejemplo
oficial. Las salidas son `loc`, `conf` e `iou`: las posiciones 4:8 y 10:14 de
`loc` definen las cuatro esquinas. La puntuación es
`sqrt(conf[:, 1] * clip(iou[:, 0], 0, 1))`. NMS utiliza cajas envolventes xywh.

El modelo fue entrenado con placas chinas; inferencia correcta no garantiza
precisión en Costa Rica. No incluye OCR ni reconocimiento de matrículas.
