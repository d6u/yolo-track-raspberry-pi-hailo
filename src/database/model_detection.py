from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from .database import Base
from .mixins import MixinCreatedAt
from .model_video_file import VideoFile


class Detection(Base, MixinCreatedAt):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))

    video_file_id: Mapped[int] = mapped_column(
        ForeignKey("video_files.id", ondelete="CASCADE"),
        index=True,
    )
    video_file: Mapped[VideoFile] = relationship(
        foreign_keys=[video_file_id],
        back_populates="detections",
    )

    def __repr__(self) -> str:
        return f"Detection(id={self.id!r}, name={self.name!r})"
