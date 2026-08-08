# Written for Python 3.12, FastAPI with Pydantic v2, and SQLAlchemy 2.x

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from sample import sample


# -----------------------------
# Database configuration
# -----------------------------

DATABASE_URL = "sqlite:///./tinygpt.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class GenerationRecord(Base):
    __tablename__ = "generation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text)
    checkpoint_path: Mapped[str] = mapped_column(String(500))
    encoding: Mapped[str] = mapped_column(String(100))
    response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


Base.metadata.create_all(engine)


# -----------------------------
# Request and response formats
# -----------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    checkpoint_path: str = "checkpoint.pt"
    encoding: str = "gpt2"


class GenerateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prompt: str
    response: str
    checkpoint_path: str
    encoding: str
    created_at: datetime


# -----------------------------
# FastAPI application
# -----------------------------

app = FastAPI(
    title="fooTinyGPT API",
    version="1.0.0",
)


def get_database():
    database = SessionLocal()

    try:
        yield database
    finally:
        database.close()


@app.get("/")
def home():
    return {"message": "fooTinyGPT API is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    database: Session = Depends(get_database),
):
    try:
        generated_text = sample(
            prompt=request.prompt,
            checkpoint_file=request.checkpoint_path,
            encoding=request.encoding,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {request.checkpoint_path}",
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {error}",
        )

    record = GenerationRecord(
        prompt=request.prompt,
        checkpoint_path=request.checkpoint_path,
        encoding=request.encoding,
        response=generated_text,
    )

    database.add(record)
    database.commit()
    database.refresh(record)

    return record


@app.get("/history", response_model=list[GenerateResponse])
def get_history(database: Session = Depends(get_database)):
    statement = select(GenerationRecord).order_by(
        GenerationRecord.created_at.desc()
    )

    return database.scalars(statement).all()


@app.get("/history/{record_id}", response_model=GenerateResponse)
def get_history_item(
    record_id: int,
    database: Session = Depends(get_database),
):
    record = database.get(GenerationRecord, record_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")

    return record