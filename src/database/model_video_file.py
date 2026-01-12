from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy import String
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from .database import Base
from .mixins import MixinCreatedAt

if TYPE_CHECKING:
    from .model_detection import Detection


class VideoFile(Base, MixinCreatedAt):
    __tablename__ = "video_files"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(30))

    detections: WriteOnlyMapped[Detection] = relationship(
        back_populates="video_file",
        cascade="all, delete",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"VideoFile(id={self.id!r}, path={self.path!r}"
