from __future__ import annotations

import asyncio
import builtins
from datetime import UTC, datetime, timedelta

from llmfy.exception.llmfy_exception import LLMfyException
from llmfy.flow_engine.checkpointer.base_checkpointer import (
    BaseCheckpointer,
    Checkpoint,
    CheckpointMetadata,
)
from llmfy.flow_engine.checkpointer.codec import DEFAULT_MAX_STATE_BYTES, StateCodec

try:
    from sqlalchemy import (
        JSON,
        Column,
        DateTime,
        Index,
        Integer,
        LargeBinary,
        String,
        TypeDecorator,
        create_engine,
        delete,
        select,
    )
    from sqlalchemy.dialects import mysql, postgresql
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker

    SQLALCHEMY_AVAILABLE = True

    class TimestampMilliseconds(TypeDecorator):
        impl = DateTime
        cache_ok = True

        def load_dialect_impl(self, dialect):
            if dialect.name == "mysql":
                return dialect.type_descriptor(mysql.DATETIME(fsp=3))
            elif dialect.name == "postgresql":
                return dialect.type_descriptor(postgresql.TIMESTAMP(precision=3))
            else:
                return dialect.type_descriptor(DateTime())

    class LongBinary(TypeDecorator):
        """Unbounded binary column — holds `StateCodec`-encoded bytes
        (JSON, optionally zlib-compressed and/or Fernet-encrypted).
        MySQL's plain `BLOB` caps out at 64KB, so it needs `LONGBLOB`
        explicitly; Postgres/SQLite's default binary type has no such cap.
        """

        impl = LargeBinary
        cache_ok = True

        def load_dialect_impl(self, dialect):
            if dialect.name == "mysql":
                return dialect.type_descriptor(mysql.LONGBLOB())
            else:
                return dialect.type_descriptor(LargeBinary())

    Base = declarative_base()

    class CheckpointModel(Base):
        """SQLAlchemy model for checkpoint storage."""

        __tablename__ = "llmfy_checkpoint"

        checkpoint_id = Column(String(255), primary_key=True)
        session_id = Column(String(255), nullable=False, index=True)
        run_id = Column(String(255), nullable=False)
        created_at = Column(TimestampMilliseconds, nullable=False)
        node = Column(String(255), nullable=False)
        prev_node = Column(String(255), nullable=False)
        step = Column(Integer, nullable=False)
        updated_fields = Column(JSON, nullable=False)
        attempt = Column(Integer, nullable=True)
        dispatch_id = Column(String(255), nullable=True)
        state = Column(LongBinary, nullable=False)

        __table_args__ = (Index("idx_thread_created_at", "session_id", "created_at"),)

except ImportError:
    SQLALCHEMY_AVAILABLE = False


class SQLCheckpointer(BaseCheckpointer):
    """
    SQL database checkpoint storage backend using SQLAlchemy.

    Supports both sync and async drivers for multiple databases:
    - PostgreSQL (async: asyncpg, sync: psycopg2)
    - MySQL (async: aiomysql, sync: pymysql)
    - SQLite (async: aiosqlite, sync: built-in)
    """

    def __init__(
        self,
        connection_string: str,
        echo: bool = False,
        compress: bool = False,
        encryption_key: bytes | str | None = None,
        max_state_bytes: int | None = DEFAULT_MAX_STATE_BYTES,
        max_checkpoints_per_session: int | None = None,
        ttl_seconds: int | None = None,
    ):
        """
        Initialize the SQL database checkpointer.

        Args:
            connection_string: SQLAlchemy connection string (sync or async)
            echo: Whether to echo SQL statements (for debugging). Bound
                parameters are logged as-is, i.e. as ciphertext only if
                `encryption_key` is set — never enable this in production.
            compress: zlib-compress the serialized state before writing.
                Off by default. When off (and `encryption_key` is unset),
                the `state` column holds plain, inspectable JSON bytes
                rather than a compressed binary blob.
            encryption_key: optional Fernet key (see
                `cryptography.fernet.Fernet.generate_key()`) to encrypt state
                at rest. Requires `pip install "llmfy[crypto]"`. `None`
                (default) stores state unencrypted.
            max_state_bytes: reject a checkpoint whose serialized state
                exceeds this many bytes, raising
                `CheckpointPayloadTooLargeException`, instead of writing an
                unbounded payload. `None` disables the check.
            max_checkpoints_per_session: if set, only the newest N
                checkpoints are retained per `session_id` — older ones are
                deleted right after each save.
            ttl_seconds: if set, checkpoints older than this age are deleted
                right after each save. Independent of, and combinable with,
                `max_checkpoints_per_session`.

        Example connection strings:

            ASYNC (Recommended):
            - PostgreSQL: "postgresql+asyncpg://user:password@localhost:5432/dbname"
            - MySQL:      "mysql+aiomysql://user:password@localhost:3306/dbname"
            - SQLite:     "sqlite+aiosqlite:///./database.db"

            SYNC (For compatibility with pymysql, psycopg2, etc):
            - PostgreSQL: "postgresql+psycopg2://user:password@localhost:5432/dbname"
            - MySQL:      "mysql+pymysql://user:password@localhost:3306/dbname"
            - SQLite:     "sqlite:///./database.db"

        Installation:
            Async drivers (recommended):
            - pip install sqlalchemy asyncpg --break-system-packages      # PostgreSQL
            - pip install sqlalchemy aiomysql --break-system-packages     # MySQL
            - pip install sqlalchemy aiosqlite --break-system-packages    # SQLite

            Sync drivers (for compatibility):
            - pip install sqlalchemy psycopg2-binary --break-system-packages  # PostgreSQL
            - pip install sqlalchemy pymysql --break-system-packages          # MySQL
            - pip install sqlalchemy --break-system-packages                  # SQLite (built-in)
        """
        if not SQLALCHEMY_AVAILABLE:
            raise LLMfyException(
                "SQLAlchemy is required for SQLCheckpointer.\n"
                "Install with: pip install sqlalchemy --break-system-packages\n\n"
                "Then install your database driver:\n"
                "  Async (recommended):\n"
                "    pip install asyncpg --break-system-packages      # PostgreSQL\n"
                "    pip install aiomysql --break-system-packages     # MySQL\n"
                "    pip install aiosqlite --break-system-packages    # SQLite\n\n"
                "  Sync (for compatibility):\n"
                "    pip install psycopg2-binary --break-system-packages  # PostgreSQL\n"
                "    pip install pymysql --break-system-packages          # MySQL\n"
                "    (SQLite is built-in, no extra package needed)"
            )

        self.connection_string = connection_string

        # Detect if using async or sync driver
        self.is_async = any(
            driver in connection_string.lower()
            for driver in ["asyncpg", "aiomysql", "aiosqlite", "+async"]
        )

        if self.is_async:
            # Async mode
            self.engine = create_async_engine(connection_string, echo=echo)
            self.session_maker = async_sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        else:
            # Sync mode
            self.engine = create_engine(connection_string, echo=echo)
            self.session_maker = sessionmaker(bind=self.engine)

        self._codec = StateCodec(
            compress=compress,
            encryption_key=encryption_key,
            max_state_bytes=max_state_bytes,
        )
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self.ttl_seconds = ttl_seconds
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure database tables are created."""
        if not self._initialized:
            if self.is_async:
                async with self.engine.begin() as conn:  # type: ignore
                    await conn.run_sync(Base.metadata.create_all)
            else:
                # Sync initialization - run in executor to keep it async
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, Base.metadata.create_all, self.engine)

            self._initialized = True

    def _retention_statements(self, session_id: str) -> list:
        """Build the DELETE statements (if any) needed to enforce this
        instance's retention config for one session, scoped to the existing
        `idx_thread_created_at` index. Empty when neither is configured, so
        callers not opting into retention pay zero extra query cost.

        `synchronize_session=False`: these deletes never need to reconcile
        with in-memory ORM objects (the caller doesn't keep references past
        `save()`), and skipping it avoids SQLAlchemy's local "evaluate"
        strategy, which would otherwise compare a freshly-committed (and,
        on sync sessions, commit-expired-then-reloaded) attribute against
        our bind parameter in Python — a real hazard here since SQLite
        round-trips datetimes as naive, which can't be compared to the
        timezone-aware `cutoff` below.
        """
        stmts = []
        if self.max_checkpoints_per_session is not None:
            keep_ids = (
                select(CheckpointModel.checkpoint_id)
                .where(CheckpointModel.session_id == session_id)
                .order_by(CheckpointModel.created_at.desc())
                .limit(self.max_checkpoints_per_session)
            )
            stmts.append(
                delete(CheckpointModel)
                .where(
                    CheckpointModel.session_id == session_id,
                    CheckpointModel.checkpoint_id.notin_(keep_ids),
                )
                .execution_options(synchronize_session=False)
            )
        if self.ttl_seconds is not None:
            cutoff = datetime.now(UTC) - timedelta(seconds=self.ttl_seconds)
            stmts.append(
                delete(CheckpointModel)
                .where(
                    CheckpointModel.session_id == session_id,
                    CheckpointModel.created_at < cutoff,
                )
                .execution_options(synchronize_session=False)
            )
        return stmts

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to SQL database."""
        await self._ensure_initialized()

        session_id = checkpoint.metadata.session_id
        model = CheckpointModel(
            checkpoint_id=checkpoint.metadata.checkpoint_id,
            session_id=session_id,
            run_id=checkpoint.metadata.run_id,
            created_at=checkpoint.metadata.created_at,
            node=checkpoint.metadata.node,
            prev_node=checkpoint.metadata.prev_node,
            step=checkpoint.metadata.step,
            updated_fields=checkpoint.metadata.updated_fields,
            attempt=checkpoint.metadata.attempt,
            dispatch_id=checkpoint.metadata.dispatch_id,
            state=self._codec.encode(checkpoint.state, session_id=session_id),
        )

        if self.is_async:
            async with self.session_maker() as session:  # type: ignore
                session.add(model)
                await session.commit()
                stmts = self._retention_statements(session_id)
                if stmts:
                    for stmt in stmts:
                        await session.execute(stmt)
                    await session.commit()
        else:
            # Sync operation - run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_sync, model)

    def _save_sync(self, model: CheckpointModel):
        """Helper for sync save."""
        with self.session_maker() as session:  # type: ignore
            session.add(model)
            session.commit()
            stmts = self._retention_statements(model.session_id)  # type: ignore[arg-type]
            if stmts:
                for stmt in stmts:
                    session.execute(stmt)
                session.commit()

    async def load(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        """Load a checkpoint from SQL database."""
        await self._ensure_initialized()

        if self.is_async:
            return await self._load_async(session_id, checkpoint_id)
        else:
            # Sync operation - run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._load_sync, session_id, checkpoint_id
            )

    async def _load_async(
        self, session_id: str, checkpoint_id: str | None
    ) -> Checkpoint | None:
        """Helper for async load."""
        async with self.session_maker() as session:  # type: ignore
            if checkpoint_id:
                stmt = select(CheckpointModel).where(
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.session_id == session_id,
                )
            else:
                stmt = (
                    select(CheckpointModel)
                    .where(CheckpointModel.session_id == session_id)
                    .order_by(CheckpointModel.created_at.desc())
                    .limit(1)
                )

            result = await session.execute(stmt)
            model = result.scalar_one_or_none()

            return self._model_to_checkpoint(model) if model else None

    def _load_sync(
        self,
        session_id: str,
        checkpoint_id: str | None,
    ) -> Checkpoint | None:
        """Helper for sync load."""
        with self.session_maker() as session:  # type: ignore
            if checkpoint_id:
                stmt = select(CheckpointModel).where(
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.session_id == session_id,
                )
            else:
                stmt = (
                    select(CheckpointModel)
                    .where(CheckpointModel.session_id == session_id)
                    .order_by(CheckpointModel.created_at.desc())
                    .limit(1)
                )

            result = session.execute(stmt)
            model = result.scalar_one_or_none()

            return self._model_to_checkpoint(model) if model else None

    async def list(self, session_id: str, limit: int = 10) -> builtins.list[Checkpoint]:
        """List checkpoints for a session."""
        await self._ensure_initialized()

        if self.is_async:
            return await self._list_async(session_id, limit)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._list_sync, session_id, limit)

    async def _list_async(self, session_id: str, limit: int) -> builtins.list[Checkpoint]:
        """Helper for async list."""
        async with self.session_maker() as session:  # type: ignore
            stmt = (
                select(CheckpointModel)
                .where(CheckpointModel.session_id == session_id)
                .order_by(CheckpointModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_checkpoint(model) for model in models]

    def _list_sync(self, session_id: str, limit: int) -> builtins.list[Checkpoint]:
        """Helper for sync list."""
        with self.session_maker() as session:  # type: ignore
            stmt = (
                select(CheckpointModel)
                .where(CheckpointModel.session_id == session_id)
                .order_by(CheckpointModel.created_at.desc())
                .limit(limit)
            )
            result = session.execute(stmt)
            models = result.scalars().all()

            return [self._model_to_checkpoint(model) for model in models]

    async def delete(self, session_id: str, checkpoint_id: str | None = None) -> None:
        """Delete checkpoint(s) from SQL database."""
        await self._ensure_initialized()

        if self.is_async:
            await self._delete_async(session_id, checkpoint_id)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._delete_sync, session_id, checkpoint_id
            )

    async def _delete_async(self, session_id: str, checkpoint_id: str | None):
        """Helper for async delete."""
        async with self.session_maker() as session:  # type: ignore
            if checkpoint_id:
                stmt = delete(CheckpointModel).where(
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.session_id == session_id,
                )
            else:
                stmt = delete(CheckpointModel).where(
                    CheckpointModel.session_id == session_id
                )

            await session.execute(stmt)
            await session.commit()

    def _delete_sync(self, session_id: str, checkpoint_id: str | None):
        """Helper for sync delete."""
        with self.session_maker() as session:  # type: ignore
            if checkpoint_id:
                stmt = delete(CheckpointModel).where(
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.session_id == session_id,
                )
            else:
                stmt = delete(CheckpointModel).where(
                    CheckpointModel.session_id == session_id
                )

            session.execute(stmt)
            session.commit()

    async def clear_all(self) -> None:
        """Clear all checkpoints from SQL database."""
        await self._ensure_initialized()

        if self.is_async:
            await self._clear_all_async()
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._clear_all_sync)

    async def _clear_all_async(self):
        """Helper for async clear_all."""
        async with self.session_maker() as session:  # type: ignore
            stmt = delete(CheckpointModel)
            await session.execute(stmt)
            await session.commit()

    def _clear_all_sync(self):
        """Helper for sync clear_all."""
        with self.session_maker() as session:  # type: ignore
            stmt = delete(CheckpointModel)
            session.execute(stmt)
            session.commit()

    def _model_to_checkpoint(self, model: CheckpointModel) -> Checkpoint:
        """Convert SQLAlchemy model to Checkpoint object."""
        metadata = CheckpointMetadata(
            checkpoint_id=model.checkpoint_id,  # type: ignore
            session_id=model.session_id,  # type: ignore
            run_id=model.run_id,  # type: ignore
            created_at=model.created_at,  # type: ignore
            node=model.node,  # type: ignore
            prev_node=model.prev_node,  # type: ignore
            step=model.step,  # type: ignore
            updated_fields=model.updated_fields,  # type: ignore
            attempt=model.attempt,  # type: ignore
            dispatch_id=model.dispatch_id,  # type: ignore
        )
        state = self._codec.decode(model.state)  # type: ignore[arg-type]
        return Checkpoint(metadata=metadata, state=state)

    async def close(self) -> None:
        """Close the database connection."""
        if self.is_async:
            await self.engine.dispose()  # type: ignore
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.engine.dispose)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
