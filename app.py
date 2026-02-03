
# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_cors import CORS
import config
from flask_jwt_extended import JWTManager
from routes.auth_routes import auth_bp
from routes.measurement_routes import measurement_bp

app = Flask(__name__)
app.config.from_object('config')
jwt = JWTManager(app)
CORS(app, resources=config.CORS_RESOURCES, supports_credentials=config.CORS_SUPPORTS_CREDENTIALS, allow_headers=config.CORS_ALLOW_HEADERS, methods=config.CORS_METHODS)
app.register_blueprint(auth_bp)
app.register_blueprint(measurement_bp)  


# Handle JWT missing/invalid token errors gracefully
from flask_jwt_extended.exceptions import NoAuthorizationError
from flask import jsonify

@app.errorhandler(NoAuthorizationError)
def handle_no_auth_error(e):
    return jsonify({'error': 'Missing or invalid Authorization header.'}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)