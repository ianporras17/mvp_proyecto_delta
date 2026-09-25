# Proyecto Delta — MVP MPC Sprint 1

MVP funcional de infraestructura MPC para video. Un cliente confiable reduce un
MP4 local a grayscale, como máximo 320x240 y 5 FPS, introduce sus pixels por el
`ExternalIO` oficial de MP-SPDZ y recibe el resultado de un programa identidad.
El cliente reconstruye el resultado en:

```text
output/<video_id>/reconstructed.mp4
```

El MP4 original y el resultado no anonimizado nunca se envían al Coordinator ni
a una party individual. Computer Vision se implementará en otro sprint.

## Arquitectura

```text
Cliente confiable
  ├─ metadata/estado HTTP ───────────────> Coordinator FastAPI
  ├─ control HTTP ───────────────────────> Node Agents P0/P1/P2
  └─ pixels uint8 privados por ExternalIO
          ├──────────────────────────────> replicated-ring Party 0
          ├──────────────────────────────> replicated-ring Party 1
          └──────────────────────────────> replicated-ring Party 2
                                             │
                           delta_video_identity.mpc
                                             │
                         reveal_to_clients (solo al cliente)
                                             ▼
                              reconstructed.mp4 local
```

Cada servicio `party-N` contiene un Node Agent FastAPI y un proceso real
`replicated-ring-party.x`. FastAPI solo controla el job; MP-SPDZ realiza el
sharing, la comunicación party-to-party y el cálculo secreto. El Node Agent
inicia o reinicia el runtime al preparar un job y lo detiene durante `cleanup`;
así un nuevo job funciona incluso después del timeout inactivo de ExternalIO.

Decisiones del MVP:

- exactamente tres parties;
- replicated secret sharing sobre ring de 64 bits;
- modelo semi-honesto de desarrollo (`program.active = False`);
- programa propio `delta_video_identity.mpc`;
- bloques de 4096 pixels;
- un único job activo;
- estado y manifests solo en memoria;
- certificados autofirmados de desarrollo;
- sin archivos persistentes de shares ni staging de video.

## Requisitos

- Docker Desktop/Engine con Docker Compose;
- el clon de MP-SPDZ como hermano del repositorio:

  ```text
  ProyectoDelta/
  ├── mvp_proyecto_delta/
  └── MP-SPDZ/
  ```

- Python 3.10 o posterior si se ejecutará la CLI en el host.

La primera construcción compila `replicated-ring-party.x` y puede tardar varios
minutos. Las reconstrucciones posteriores usan la caché de Docker.

## Construir y levantar

Desde `mvp_proyecto_delta/`:

```bash
docker compose up --build -d
docker compose ps
```

Compose levanta:

| Servicio | Puerto host | Responsabilidad |
|---|---:|---|
| `coordinator-api` | 8000 | UUID, metadata y ciclo del job |
| `party-0` | 8100 / 14000 | Node Agent / ExternalIO P0 |
| `party-1` | 8101 / 14001 | Node Agent / ExternalIO P1 |
| `party-2` | 8102 / 14002 | Node Agent / ExternalIO P2 |
| `cert-init` | — | genera certificados de desarrollo y termina |

Los runtimes usan internamente `party-0:5000`, `party-1:5001` y
`party-2:5002`. `docker compose ps` debe mostrar saludables al Coordinator y a
las tres parties.

## Ejecutar un video

### Opción A: CLI en el host

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m client upload C:\ruta\video.mp4
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m client upload /ruta/video.mp4
```

El comando equivalente instalado es:

```bash
delta upload /ruta/video.mp4
```

### Opción B: cliente confiable en Docker

Si el video se encuentra dentro de la carpeta del repositorio:

```bash
docker compose run --rm delta-client upload /workspace/video.mp4
```

Solo `delta-client` monta el repositorio que contiene el video. Coordinator y
parties no tienen ese volumen.

## Flujo y validaciones

La CLI:

1. valida el MP4 y el límite de 20 segundos;
2. reduce localmente resolución/FPS y convierte a grayscale `uint8`;
3. registra metadata pública y obtiene un UUID;
4. prepara los tres Node Agents;
5. envía pixels privados a los tres runtimes mediante ExternalIO;
6. ejecuta identidad y revela cada bloque exclusivamente al cliente;
7. compara exactamente los bytes enviados y recibidos antes de codificar;
8. escribe y vuelve a abrir `reconstructed.mp4`;
9. marca `COMPLETED` y limpia el estado temporal de las parties.

Estados del Coordinator:

```text
REGISTERED → PREPARING → MPC_READY → PROCESSING → COMPLETED
                                             └─→ FAILED
```

Un segundo registro mientras el primero no se haya limpiado recibe `409`.

## Pruebas

Instalar dependencias de desarrollo y ejecutar la suite rápida:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -m "not e2e"
```

Con Compose saludable, ejecutar el smoke test real (64x48, 2 FPS, 2 s):

PowerShell:

```powershell
$env:RUN_MPC_E2E="1"
python -m pytest tests/integration/test_mpc_e2e.py -s
```

Linux/macOS:

```bash
RUN_MPC_E2E=1 python -m pytest tests/integration/test_mpc_e2e.py -s
```

## Apagar y limpiar

```bash
docker compose down
```

Los jobs no crean shares ni archivos de entrada en las parties. `.runtime/`
conserva certificados de desarrollo reutilizables y logs mínimos de los
runtimes. Los resultados permanecen bajo `output/` en el cliente.

## Limitaciones del Sprint 1

- el cálculo es identidad; todavía no hay rostros, placas, blur ni modelos;
- las tres parties viven en un solo host y una misma organización;
- los certificados son de desarrollo y la imagen compartida puede ver el mismo
  directorio de certificados;
- no hay base de datos ni recuperación tras reiniciar el Coordinator;
- se procesa un video a la vez;
- perfil máximo: 20 s, 320x240, grayscale, 5 FPS;
- no se ha preparado todavía el despliegue ARM/Raspberry Pi.
