"""Modelos del plano de control del MVP MPC."""

from enum import Enum
from typing import Dict, List, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class VideoStatus(str, Enum):
    """Ciclo de vida de un único job MPC."""

    REGISTERED = "REGISTERED"
    PREPARING = "PREPARING"
    MPC_READY = "MPC_READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PartyStatus(str, Enum):
    """Estado de control de una computing party."""

    PENDING = "PENDING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PartyEndpoint(BaseModel):
    """Endpoint público del Node Agent de una party."""

    id: int
    url: str


class VideoMetadata(BaseModel):
    """Metadata pública mínima de la representación MPC local."""

    width: int = Field(ge=1, le=320)
    height: int = Field(ge=1, le=240)
    fps: float = Field(gt=0, le=5)
    frame_count: int = Field(ge=1, le=100)
    dtype: Literal["uint8"] = "uint8"
    channels: Literal[1] = 1
    version: Literal[2] = 2


class VideoRegistrationRequest(BaseModel):
    """Metadata no sensible usada para registrar un video."""

    filename: str = Field(min_length=1, max_length=255)
    format: str = Field(min_length=1, max_length=20)


class VideoPrepareRequest(BaseModel):
    """Representación que las parties deben esperar por ExternalIO."""

    metadata: VideoMetadata


class VideoRegistrationResponse(BaseModel):
    """Job nuevo y endpoints de control de las tres parties."""

    id: UUID
    status: VideoStatus
    parties: List[PartyEndpoint]


class VideoStateResponse(BaseModel):
    """Snapshot del job sin contener pixels ni shares."""

    id: UUID
    status: VideoStatus
    parties: Dict[str, PartyStatus]
    metadata: VideoMetadata | None = None
    error: str | None = None


class PartyReadyRequest(BaseModel):
    """Callback emitido por un Node Agent listo."""

    status: Literal["READY"] = "READY"


class FailureRequest(BaseModel):
    """Detalle operacional de un job fallido."""

    detail: str = Field(min_length=1, max_length=1000)
