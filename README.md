# Proyecto Delta — MVP de anonimización con MPC

El cliente detecta **rostros con YuNet y placas con LPD-YuNet**, a color y en
la resolución original de cada cuadro seleccionado. Las tres parties reales
de MP-SPDZ reciben los píxeles privados y ocultan las regiones detectadas con
**rectángulos negros**. El cliente reconstruye únicamente la salida del cálculo.

No hay OCR, identificación de personas, blur ni seguimiento temporal.
Un detector puede omitir regiones o producir falsos positivos: este MVP no
garantiza que desaparezca toda información sensible del video.

## Arquitectura

1. Cliente: abre el MP4, muestrea hasta 10 FPS y ejecuta ambos detectores.
2. Cliente: ajusta las cajas al tamaño MPC y prepara máscaras públicas.
3. Coordinator y Node Agents: coordinan el trabajo mediante HTTP; no reciben el MP4.
4. ExternalIO: introduce los píxeles originales como valores secretos.
5. MP-SPDZ: calcula `salida = píxel_secreto × (1 − máscara_pública)`.
6. Cliente: valida ceros dentro de las cajas e igualdad fuera, y codifica el MP4.

La máscara y las posiciones son públicas; los píxeles de entrada permanecen
secretos. Los modelos corren en el cliente, **no dentro de MP-SPDZ**.
Las tres parties usan replicated-ring de 64 bits y el modelo semi-honesto.

## Requisitos y estructura

Docker Desktop/Engine con Compose y el clon de MP-SPDZ como hermano:

```text
ProyectoDelta/
├── MP-SPDZ/
├── mvp_proyecto_delta/
├── video2.mp4
└── video3.mp4
```

El repositorio general proyecto-delta no se necesita para construir o ejecutar
este MVP. Ambos ONNX y sus licencias están incluidos; no se descargan modelos
durante la inferencia. Para la CLI local se necesita Python 3.10 o posterior.

## Preparar el cliente local

Desde mvp_proyecto_delta, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Construir y levantar

```powershell
docker compose up --build -d
docker compose ps
```

El coordinador y las tres parties deben aparecer saludables. cert-init genera
o reutiliza certificados de desarrollo y termina; Exited (0) es normal.
Si están vencidos, dañados o vencen en menos de 24 horas, los renueva antes
de arrancar las parties. No hace falta borrar volúmenes para resolver su vencimiento.
La primera construcción compila MP-SPDZ y puede tardar varios minutos.

No levantar a la vez el Compose de proyecto-delta: ambos publican los mismos
puertos. Detener primero el otro Compose si está activo.

| Servicio | Puerto del host | Función |
|---|---:|---|
| coordinator-api | 8000 | Registro y estado global |
| party-0 / party-1 / party-2 | 8100 / 8101 / 8102 | APIs de control |
| ExternalIO de las tres parties | 14000 / 14001 / 14002 | Entrada privada y resultado MPC |

Las parties se comunican dentro de Docker por 5000/5001/5002.
Las APIs publican su documentación en /docs.

## Probar rostros y placas juntos

Con el entorno activado y Compose saludable:

```powershell
python -m client upload "..\video3.mp4" --processing-mode anonymize --detect-faces --detect-plates
```

El resultado se imprime en consola y queda en:

```text
output/<video_id>/reconstructed.mp4
```

También pueden ejecutarse por separado, usando solo --detect-faces o solo
--detect-plates. Los umbrales predeterminados son 0,6; el adaptador de rostros
aplica además un mínimo fijo estrictamente mayor que 0,6, aunque se configure
un umbral menor en OpenCV.
--plate-confidence-threshold permite configurar placas y --conf-threshold,
YuNet. Una confianza alta no es un porcentaje de precisión comprobado.

**Sin --processing-mode anonymize, upload conserva identidad y no anonimiza.**
Ese modo permite comprobar la infraestructura anterior:

```powershell
python -m client upload "..\video3.mp4" --processing-mode identity
```

El programa predeterminado de las parties, delta_video_anonymize, admite ambos
modos. Se conserva delta_video_identity como referencia compatible.
--regions-json es una entrada manual opcional; no se combina con los detectores.

### Perfil de video

Hasta 20 segundos, 10 FPS, gris y caja máxima 640×480; se conservan proporciones
y no se amplían videos pequeños. Un video vertical puede resultar, por ejemplo,
en 268×480: 640×480 es un límite, no un tamaño forzado.
Los cuadros de detección siguen siendo los originales BGR antes del resize/gris.

Para una prueba de menor costo, sin cambiar los valores predeterminados:

```powershell
python -m client upload "..\video3.mp4" --processing-mode anonymize --detect-faces --detect-plates --mpc-max-width 320 --mpc-max-height 240 --mpc-fps 5
```

La resolución de los servicios se configura con MPC_MAX_WIDTH y MPC_MAX_HEIGHT
en Compose. La CLI admite esas variables o los argumentos mostrados.
El archivo .env.example es una referencia; la CLI local no carga .env sola.
Aceptar el máximo no demuestra tiempo real ni viabilidad de 20 segundos a
640×480/10 FPS.

### Cliente opcional en Docker

Si el video está dentro de este repositorio:

```powershell
docker compose run --rm delta-client upload /workspace/video.mp4 --processing-mode anonymize --detect-faces --detect-plates
```

Solo el cliente monta el repositorio con los videos. Las parties y el
coordinador no tienen ese montaje.

## Archivos principales

| Ruta | Responsabilidad |
|---|---|
| src/client/cli.py | Flujo completo y selección de detectores/modo |
| src/client/preprocessing/ | Muestreo, YuNet y reconstrucción |
| src/client/detection/ | LPD-YuNet y recurso del modelo |
| src/client/regions.py | Validación y transformación de cajas |
| src/client/anonymization.py | Plan y máscaras públicas; no oculta píxeles localmente |
| src/mpc/external_io.py | Transporte privado por bloques de 4096 |
| mpc/Programs/Source/delta_video_anonymize.mpc | Ocultamiento dentro de MPC |
| src/coordinator/ | API de estado global |
| src/party_receiver/ | Node Agent y supervisor del runtime |
| docker-compose.yml / docker/mp-spdz.Dockerfile | Arranque e imagen de las parties |

YuNet conserva su [licencia MIT](src/client/preprocessing/LICENSE).
LPD-YuNet incluye [licencia Apache 2.0 y procedencia](src/client/detection/models/README.md).
No se incluyen investigaciones, demostradores, videos ni salidas generadas.

## Pruebas

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -m "not e2e" -q
$env:RUN_MPC_E2E="1"
python -m pytest -m e2e -q
```

Los E2E requieren Compose saludable y verifican el resultado real antes del
códec MP4. Se comprueban identidad, anonimización de cajas conocidas, orden,
dimensiones, FPS, fallos y limpieza. No se utiliza una respuesta simulada en
lugar del resultado de MP-SPDZ.

### Verificación de entrega — 23 de octubre de 2026

Se ejecutó el flujo real con ambos detectores y tres parties en los dos
repositorios, usando `../video3.mp4`: los primeros tres segundos de `video2.mp4`
(90 cuadros originales, a color, de 478×850). El original permaneció intacto.
El perfil MPC produjo 30 cuadros de 268×480, a 10 FPS y en gris; los detectores
recibieron los cuadros seleccionados en su resolución y color originales.
Se obtuvieron 28 observaciones de rostros y 29 de placas, no identidades únicas
ni una medida de precisión.

| Ejecución real | ExternalIO/MPC | Flujo completo |
|---|---:|---:|
| proyecto-delta | 271,328 s | 288,946 s |
| mvp_proyecto_delta | 285,821 s | 301,966 s |

ExternalIO incluye el cliente, el transporte y las esperas de las parties;
no es una medición exclusiva del cálculo C++. Se comprobó antes del códec que
los píxeles dentro de las cajas fueran cero y los exteriores permanecieran
idénticos. Ambos repositorios produjeron los mismos hashes de entrada, máscara,
resultado MPC y MP4. También se verificaron los 30 cuadros del MP4 y la limpieza
de los trabajos. Esto no demuestra tiempo real ni 20 segundos a 640×480.

Pasaron 544 pruebas rápidas y tres E2E reales en Delta. En este MVP pasaron
484 pruebas, incluidos sus cuatro E2E reales. También se verificó que los dos
modelos y sus licencias se instalan y permiten inferencia sin conexión a Internet.
La evidencia local está en `.runtime/delivery/`; no forma parte del producto.
Los certificados de desarrollo vencidos se respaldaron y renovaron; el arranque
ahora verifica su vigencia antes de reutilizarlos.

## Apagar

```powershell
docker compose down
```

Los resultados permanecen en output/ y los certificados/logs en .runtime/.
Ambas carpetas están ignoradas por Git. Los estados de jobs son en memoria,
sin base de datos ni recuperación después de reiniciar.

## Límites de seguridad y despliegue

- Tres parties en una computadora: no representan tres organizaciones independientes.
- Certificados autofirmados de desarrollo, con acceso compartido en el entorno local.
- El cliente oficial usa TLS, pero no verifica el certificado del servidor
  ni su hostname en su configuración actual; no se afirma autenticación mutua.
- Las APIs usan HTTP y no implementan autenticación/autorización.
- No exponer este despliegue de desarrollo a una red pública sin endurecerlo.
- Se procesa un trabajo a la vez; detección y anonimización no garantizan cobertura total.
- Los modelos analizan los cuadros seleccionados, no todos los cuadros descartados por muestreo.
- El MP4 se recodifica con pérdidas y sin conservar el audio.
- No se ha validado despliegue ARM/Raspberry Pi ni procesamiento en tiempo real.
