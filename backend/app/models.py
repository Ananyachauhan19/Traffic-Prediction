from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import json

from pydantic import BaseModel

from sqlalchemy import (
	create_engine,
	Column,
	Integer,
	Float,
	String,
	ForeignKey,
	DateTime,
	Text,
	func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session


Base = declarative_base()


class RoadSegment(Base):
	__tablename__ = 'road_segments'
	id = Column(Integer, primary_key=True)
	name = Column(String(200), unique=True, nullable=False)
	width = Column(Float, nullable=False)
	# store coordinates as JSON string (list of [lat,lng] pairs)
	coordinates = Column(Text, nullable=True)

	records = relationship('TrafficRecord', back_populates='segment', cascade='all, delete-orphan')

	def set_coordinates(self, coords: List[Tuple[float, float]]):
		self.coordinates = json.dumps(coords)

	def get_coordinates(self) -> Optional[List[Tuple[float, float]]]:
		if not self.coordinates:
			return None
		return json.loads(self.coordinates)


class TrafficRecord(Base):
	__tablename__ = 'traffic_records'
	id = Column(Integer, primary_key=True)
	segment_id = Column(Integer, ForeignKey('road_segments.id'), nullable=True)
	entry_count = Column(Integer, nullable=False)
	exit_count = Column(Integer, nullable=False)
	avg_speed = Column(Float, nullable=True)
	time_of_day = Column(Float, nullable=True)
	weather = Column(Float, nullable=True)
	congestion_level = Column(Integer, nullable=True)
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

	segment = relationship('RoadSegment', back_populates='records')


# --- DB helpers ---


def init_engine(sqlite_path: str):
	"""Create SQLAlchemy engine for given sqlite file path."""
	engine = create_engine(f'sqlite:///{sqlite_path}', connect_args={'check_same_thread': False})
	return engine


def create_tables(engine):
	Base.metadata.create_all(engine)


def get_session(engine) -> Session:
	SessionLocal = sessionmaker(bind=engine)
	return SessionLocal()


def get_or_create_segment(session: Session, name: str, width: float, coords: Optional[List[Tuple[float, float]]] = None) -> RoadSegment:
	seg = session.query(RoadSegment).filter_by(name=name).one_or_none()
	if seg:
		return seg
	seg = RoadSegment(name=name, width=width)
	if coords:
		seg.set_coordinates(coords)
	session.add(seg)
	session.commit()
	session.refresh(seg)
	return seg


def insert_traffic_record(session: Session, segment_id: Optional[int], entry_count: int, exit_count: int, avg_speed: float, time_of_day: float, weather: float, congestion_level: Optional[int] = None) -> TrafficRecord:
	rec = TrafficRecord(
		segment_id=segment_id,
		entry_count=entry_count,
		exit_count=exit_count,
		avg_speed=avg_speed,
		time_of_day=time_of_day,
		weather=weather,
		congestion_level=congestion_level,
	)
	session.add(rec)
	session.commit()
	session.refresh(rec)
	return rec


def get_recent_averages(session: Session, segment_id: int = None, window_minutes: int = 60):
	"""Return recent averages (entry_count, exit_count, avg_speed, congestion_level) over the last window_minutes."""
	cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
	q = session.query(
		func.avg(TrafficRecord.entry_count).label('avg_entry'),
		func.avg(TrafficRecord.exit_count).label('avg_exit'),
		func.avg(TrafficRecord.avg_speed).label('avg_speed'),
		func.avg(TrafficRecord.congestion_level).label('avg_congestion'),
		func.count(TrafficRecord.id).label('n_records'),
	).filter(TrafficRecord.created_at >= cutoff)
	if segment_id is not None:
		q = q.filter(TrafficRecord.segment_id == segment_id)
	row = q.one()
	return {
		'avg_entry': float(row.avg_entry) if row.avg_entry is not None else 0.0,
		'avg_exit': float(row.avg_exit) if row.avg_exit is not None else 0.0,
		'avg_speed': float(row.avg_speed) if row.avg_speed is not None else 0.0,
		'avg_congestion': float(row.avg_congestion) if row.avg_congestion is not None else 0.0,
		'n_records': int(row.n_records),
	}


# Keep Pydantic models for API validation


class TrafficIngest(BaseModel):
	entry_count: int
	exit_count: int
	road_width: float
	time_of_day: float  # e.g., hour in decimal or normalized
	weather: float      # numeric encoding for weather
	avg_speed: float


class PredictionResponse(BaseModel):
	status: str
	probability: float
	alternate_route: Optional[dict] = None

