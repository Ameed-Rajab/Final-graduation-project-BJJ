from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.analysis import Analysis
from extensions import db
from services.ai_analyzer import BJJAnalyzer
from services.video_processor import VideoProcessor
import os
import threading
import json
import requests
from collections import Counter

analysis_bp = Blueprint('analysis', __name__)


def build_local_analysis_payload(file_path):
    processor = VideoProcessor(target_fps=2, max_frames=120)
    frames, duration, original_fps = processor.extract_frames(file_path)

    if not frames:
        raise ValueError('No readable frames found in video.')

    analyzer = BJJAnalyzer()
    basic_result = analyzer.analyze_frames(frames)
    positions = basic_result.get('positions', [])

    counts = Counter(p.get('position', 'unknown') for p in positions)
    total_positions = max(len(positions), 1)
    top_position_families = []

    for family, count in counts.most_common(6):
        top_position_families.append({
            'family': family,
            'label': analyzer.POSITION_LABELS.get(family, family.replace('_', ' ').title()),
            'count': count,
            'share_percent': round((count / total_positions) * 100, 1)
        })

    segments = []
    current = None

    for position in positions:
        family = position.get('position', 'unknown')
        frame_number = position.get('frame', 0)

        if current and current['family'] == family:
            current['end_frame'] = frame_number
            current['duration_rows'] += 1
        else:
            if current:
                segments.append(current)

            current = {
                'family': family,
                'label': analyzer.POSITION_LABELS.get(family, family.replace('_', ' ').title()),
                'start_frame': frame_number,
                'end_frame': frame_number,
                'duration_rows': 1
            }

    if current:
        segments.append(current)

    longest_phases = sorted(
        segments,
        key=lambda phase: phase.get('duration_rows', 0),
        reverse=True
    )[:5]

    top_share = top_position_families[0]['share_percent'] if top_position_families else 0
    reliability_band = 'medium' if len(frames) >= 30 else 'low'

    if len(frames) >= 75:
        reliability_band = 'high'

    guard_positions = {'guard', 'closed_guard', 'open_guard', 'half_guard'}
    top_positions = {'mount', 'full_mount', 'side_control', 'north_south', 'knee_on_belly'}
    turtle_positions = {'turtle'}
    finish_positions = {'back_control', 'mount', 'full_mount'}

    def ratio(position_set):
        count = sum(counts.get(position, 0) for position in position_set)
        return round((count / total_positions) * 100, 1)

    guard_ratio = ratio(guard_positions)
    top_ratio = ratio(top_positions)
    turtle_ratio = ratio(turtle_positions)
    finish_ratio = ratio(finish_positions)

    dominant_label = (
        top_position_families[0]['label']
        if top_position_families
        else 'Unknown'
    )

    return {
        'players': [
            {
                'player_id': 1,
                'player_name': 'Player 1',
                'primary_profile': f'{dominant_label} oriented player',
                'metrics': {
                    'guard_ratio_percent': guard_ratio,
                    'top_ratio_percent': top_ratio,
                    'turtle_ratio_percent': turtle_ratio,
                    'finish_ratio_percent': finish_ratio
                },
                'secondary_traits': [
                    'Position control',
                    'Transition awareness'
                ],
                'counter_plan': {
                    'recommended_style': 'Keep posture, win inside frames, and force clean resets.',
                    'key_actions': [
                        'Control distance before entering exchanges.',
                        'Break grips before advancing position.',
                        'Stabilize after each transition.'
                    ]
                }
            },
            {
                'player_id': 2,
                'player_name': 'Player 2',
                'primary_profile': 'Reactive positional player',
                'metrics': {
                    'guard_ratio_percent': round(max(0, 100 - guard_ratio), 1),
                    'top_ratio_percent': round(max(0, 100 - top_ratio), 1),
                    'turtle_ratio_percent': round(max(0, 100 - turtle_ratio), 1),
                    'finish_ratio_percent': round(max(0, 100 - finish_ratio), 1)
                },
                'secondary_traits': [
                    'Defensive movement',
                    'Scramble response'
                ],
                'counter_plan': {
                    'recommended_style': 'Slow the pace and deny dominant grips.',
                    'key_actions': [
                        'Recover frames before accepting pressure.',
                        'Turn toward the opponent during scrambles.',
                        'Use underhooks to prevent consolidation.'
                    ]
                }
            }
        ],
        'match_overview': {
            'frames_analyzed': len(frames),
            'video_duration_seconds': round(duration, 2),
            'source_fps': round(original_fps, 2),
            'top_position_families': top_position_families
        },
        'match_sequence': {
            'segment_count': len(segments),
            'average_phase_frames': round(
                sum(phase['duration_rows'] for phase in segments) / max(len(segments), 1),
                1
            ),
            'opening_sequence': [phase['label'] for phase in segments[:3]],
            'ending_sequence': [phase['label'] for phase in segments[-3:]],
            'longest_phases': longest_phases,
            'narratives': [
                (
                    f'Analyzed {len(frames)} sampled frames from a '
                    f'{duration:.1f}s video. The most common position family was '
                    f'{dominant_label} at {top_share:.1f}% of sampled frames.'
                ),
                (
                    f'The sequence contained {len(segments)} positional phases, '
                    'giving a compact overview without sending every raw frame.'
                )
            ]
        },
        'reliability': {
            'band': reliability_band,
            'frames_used': len(frames),
            'note': 'Local sampled-frame analysis'
        },
        'source_summary': {
            'frames_used': len(frames),
            'duration_seconds': round(duration, 2),
            'source_fps': round(original_fps, 2),
            'analysis_mode': 'local_fallback'
        },
        'raw_positions': positions
    }


def run_analysis(app, analysis_id, file_path, media_type):
    with app.app_context():
        analysis = Analysis.query.get(analysis_id)

        if not analysis:
            return

        try:
            if media_type != 'video':
                raise ValueError('Only video analysis is supported.')

            ai_url = app.config.get('AI_BACKEND_URL', 'http://127.0.0.1:8000/predict')

            with open(file_path, 'rb') as f:
                response = requests.post(
                    ai_url,
                    files={'file': (os.path.basename(file_path), f, 'video/mp4')},
                    timeout=3600
                )

            if response.status_code != 200:
                if response.status_code == 422:
                    data = build_local_analysis_payload(file_path)
                else:
                    raise ValueError(f"AI backend returned {response.status_code}: {response.text}")
            else:
                data = response.json()


            analysis.positions_detected = json.dumps(data)

            analysis.frame_count = data.get("match_overview", {}).get("frames_analyzed", 0)
            analysis.duration = analysis.frame_count / 30.0 if analysis.frame_count else 0

            top_family = data.get("match_overview", {}).get("top_position_families", [])

            if top_family:
                analysis.dominant_position = top_family[0].get("family", "unknown")
            else:
                analysis.dominant_position = "unknown"

            longest = data.get("match_sequence", {}).get("longest_phases", [])

            timeline = []
            for p in longest:
                timeline.append({
                    "timestamp": float(p.get("start_frame", 0)) / 30.0,
                    "position": p.get("family", ""),
                    "label": p.get("label", ""),
                    "from_position": ""
                })

            analysis.timeline_events = json.dumps(timeline)

            narratives = data.get("match_sequence", {}).get("narratives", [])

            if narratives:
                analysis.summary = " ".join(narratives)
            else:
                analysis.summary = "Player style analysis completed."

            players = data.get("players", [])

            if len(players) >= 2:
                analysis.player1_score = players[0].get("metrics", {}).get("guard_ratio_percent", 0)
                analysis.player2_score = players[1].get("metrics", {}).get("guard_ratio_percent", 0)
            else:
                analysis.player1_score = 0
                analysis.player2_score = 0

            analysis.status = 'completed'
            db.session.commit()

        except Exception as e:
            analysis.status = 'failed'
            analysis.summary = f'Analysis failed: {str(e)}'
            db.session.commit()


@analysis_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload():
    user_id = int(get_jwt_identity())

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    title = request.form.get('title', 'BJJ Analysis')
    media_type = request.form.get('media_type', 'video')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"user_{user_id}_{len(title)}_{file.filename}"
    upload_folder = current_app.config['UPLOAD_FOLDER']

    if media_type == 'image':
        save_path = os.path.join(upload_folder, 'images', filename)
    else:
        save_path = os.path.join(upload_folder, filename)

    file.save(save_path)

    analysis = Analysis(
        user_id=user_id,
        title=title,
        media_type=media_type,
        file_path=save_path,
        status='processing'
    )

    db.session.add(analysis)
    db.session.commit()

    app = current_app._get_current_object()

    thread = threading.Thread(
        target=run_analysis,
        args=(app, analysis.id, save_path, media_type)
    )

    thread.daemon = True
    thread.start()

    return jsonify({
        'message': 'Analysis started',
        'analysis_id': analysis.id,
        'status': 'processing'
    }), 202


@analysis_bp.route('/status/<int:analysis_id>', methods=['GET'])
@jwt_required()
def status(analysis_id):
    user_id = int(get_jwt_identity())

    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    return jsonify({'analysis': analysis.to_dict()}), 200


@analysis_bp.route('/<int:analysis_id>', methods=['GET'])
@jwt_required()
def get_analysis(analysis_id):
    user_id = int(get_jwt_identity())

    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first()

    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404

    if analysis.status == 'completed' and analysis.positions_detected:
        try:
            style_json = json.loads(analysis.positions_detected)
            return jsonify({'analysis': style_json}), 200
        except Exception:
            return jsonify({'analysis': analysis.to_dict()}), 200

    return jsonify({'analysis': analysis.to_dict()}), 200
