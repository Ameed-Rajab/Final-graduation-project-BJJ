from extensions import db
from datetime import datetime
import json

class Analysis(db.Model):
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    media_type = db.Column(db.String(20), nullable=False)  # video, image
    file_path = db.Column(db.String(500))
    status = db.Column(db.String(30), default='processing')  # processing, completed, failed
    duration = db.Column(db.Float, default=0)
    frame_count = db.Column(db.Integer, default=0)
    positions_detected = db.Column(db.Text, default='[]')
    timeline_events = db.Column(db.Text, default='[]')
    summary = db.Column(db.Text, default='')
    dominant_position = db.Column(db.String(100), default='')
    player1_score = db.Column(db.Float, default=0)
    player2_score = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    thumbnail_path = db.Column(db.String(500), default='')

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'media_type': self.media_type,
            'status': self.status,
            'duration': self.duration,
            'frame_count': self.frame_count,
            'positions_detected': json.loads(self.positions_detected or '[]'),
            'timeline_events': json.loads(self.timeline_events or '[]'),
            'summary': self.summary,
            'dominant_position': self.dominant_position,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'created_at': self.created_at.isoformat(),
            'thumbnail_path': self.thumbnail_path
        }
