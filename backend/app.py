from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from extensions import db, jwt
import os
from datetime import timedelta

load_dotenv()

def create_app():
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(
        days=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES_DAYS', 7))
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///bjj.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 104857600))
    app.config['AI_BACKEND_URL'] = os.getenv('AI_BACKEND_URL', 'http://127.0.0.1:8000/predict')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'frames'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    db.init_app(app)
    jwt.init_app(app)

    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return {'error': 'Login expired. Please sign in again.'}, 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return {'error': 'Invalid login token. Please sign in again.'}, 401

    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return {'error': 'Login required. Please sign in again.'}, 401

    from routes.auth import auth_bp
    from routes.analysis import analysis_bp
    from routes.history import history_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(history_bp, url_prefix='/api/history')

    with app.app_context():
        db.create_all()

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
