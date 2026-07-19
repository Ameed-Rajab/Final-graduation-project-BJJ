from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.analysis import Analysis
from extensions import db
history_bp = Blueprint('history', __name__)

@history_bp.route('/', methods=['GET'])
@jwt_required()
def get_history():
    user_id = int(get_jwt_identity())
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    analyses = Analysis.query.filter_by(user_id=user_id)\
        .order_by(Analysis.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'analyses': [a.to_dict() for a in analyses.items],
        'total': analyses.total,
        'pages': analyses.pages,
        'current_page': page
    }), 200


@history_bp.route('/<int:analysis_id>', methods=['DELETE'])
@jwt_required()
def delete_analysis(analysis_id):
    user_id = int(get_jwt_identity())
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user_id).first()
    if not analysis:
        return jsonify({'error': 'Not found'}), 404

    db.session.delete(analysis)
    db.session.commit()
    return jsonify({'message': 'Deleted successfully'}), 200
