from datetime import datetime

from sqlalchemy import DateTime, FetchedValue, func
from sqlalchemy.orm import Mapped, mapped_column


# Having both default and server_default ensures the value is provided
# regardless how the row is inserted.
class MixinCreatedAt:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(),
        default=func.now(),
        server_default=FetchedValue(),
    )


class MixinUpdatedAt:
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(),
        default=func.now(),
        server_default=FetchedValue(),
        onupdate=func.now(),
    )
