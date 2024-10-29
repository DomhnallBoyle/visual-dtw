"""User configuration model.

Contains a db schema for the configuration table in the database
"""
from main.models import Base
from sqlalchemy import Column, Float, Integer, String


class Config(Base):
    """User Configuration model."""
    __tablename__ = 'config'
    __json__ = ['frame_step', 'delta_1', 'delta_2', 'dtw_top_n_tail',
                'dtw_transition_cost', 'dtw_beam_width', 'dtw_distance_metric',
                'knn_type', 'knn_k', 'frames_per_second']

    def __init__(self,
                 id=None,
                 frame_step=1,
                 frames_per_second=25,
                 delta_1=100,
                 delta_2=0,
                 dtw_top_n_tail=0,
                 dtw_beam_width=0,
                 dtw_distance_metric='euclidean_squared',
                 dtw_transition_cost=0.1,
                 knn_type=2,
                 knn_k=50):
        self.id = id
        self.frame_step = frame_step
        self.frames_per_second = frames_per_second
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.dtw_top_n_tail = dtw_top_n_tail
        self.dtw_beam_width = dtw_beam_width
        self.dtw_distance_metric = dtw_distance_metric
        self.dtw_transition_cost = dtw_transition_cost
        self.knn_type = knn_type
        self.knn_k = knn_k

    frame_step = Column(Integer, default=1)
    frames_per_second = Column(Integer, default=25)
    delta_1 = Column(Integer, default=100)
    delta_2 = Column(Integer, default=0)
    dtw_top_n_tail = Column(Integer, default=0)
    dtw_beam_width = Column(Integer, default=0)
    dtw_distance_metric = Column(String, default='euclidean_squared')
    dtw_transition_cost = Column(Float, default=0.1)
    knn_type = Column(Integer, default=2)
    knn_k = Column(Integer, default=50)
