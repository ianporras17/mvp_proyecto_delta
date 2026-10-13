"""Modelos del plano de control del MVP MPC."""

from enum import Enum
from typing import Dict, List, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from mpc.video_profile import MAX_CONFIGURED_FPS, MAX_VIDEO_FRAMES, MpcVideoProfile


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

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0, le=MAX_CONFIGURED_FPS)
    frame_count: int = Field(ge=1, le=MAX_VIDEO_FRAMES)
    dtype: Literal["uint8"] = "uint8"
    channels: Literal[1] = 1
    version: Literal[2] = 2

    @model_validator(mode="after")
    def validate_resolution(self) -> "VideoMetadata":
        profile = MpcVideoProfile.from_environment()
        if self.width > profile.max_width:
            raise ValueError(f"width no puede superar {profile.max_width}")
        if self.height > profile.max_height:
            raise ValueError(f"height no puede superar {profile.max_height}")
        return self


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
