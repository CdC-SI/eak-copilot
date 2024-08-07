from typing import Optional
from sqlalchemy import Integer, ForeignKey, String, Text, DateTime, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, to_tsvector
# SQLAlchemy-2.0.30

from sqlalchemy.ext.declarative import declarative_base, declared_attr


class EmbeddedMixin (object):
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(1536), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)

    url: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now())
    modified_at: Mapped[DateTime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('text_tsv', 'text', postgresql_using='gin'),
    )


Base = declarative_base()


class Source(Base):
    __tablename__ = "source"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    def __repr__(self) -> str:
        return f"Source(id={self.id!r}, url={self.url!r})"


class Document(Base, EmbeddedMixin):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    source_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("source.id"), nullable=True)
    source: Mapped[Optional["Source"]] = relationship("Source", back_populates="documents")

    __table_args__ = (
        Index('text_tsv', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}),
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, url={self.url!r}, text={self.text!r}, language={self.language!r})"


class Question(Base, EmbeddedMixin):
    __tablename__ = "question"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    answer_id: Mapped[int] = mapped_column(Integer, ForeignKey("document.id"), nullable=False)
    answer: Mapped["Document"] = relationship("Document", back_populates="questions")

    source_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("source.id"), nullable=True)
    source: Mapped[Optional["Source"]] = relationship("Source", back_populates="questions")

    __table_args__ = (
        Index('idx_text_gin', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}),
    )

    def __repr__(self) -> str:
        return f"Question(id={self.id!r}, url={self.url!r}, question={self.text!r}, answer_id={self.answer_id!r}, language={self.language!r})"


# Init relationship mappers
Document.questions = relationship("Question", order_by=Question.id, back_populates="answer")
Question.source = relationship("Source", back_populates="questions")
Source.questions = relationship("Question", order_by=Question.id, back_populates="source")
Source.documents = relationship("Document", order_by=Document.id, back_populates="source")
