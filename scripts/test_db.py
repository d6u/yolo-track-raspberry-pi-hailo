from sqlalchemy.orm import Session

from src.database.database import Base, engine
from src.database.model_detection import Detection

Base.metadata.create_all(engine)

# with Session(engine) as session:
#     d1 = Detection(name="cat")
#     session.add_all([d1])
#     session.commit()
