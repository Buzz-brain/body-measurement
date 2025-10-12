import os
from datetime import timedelta

JWT_SECRET = os.environ.get('JWT_SECRET', 'dev_secret_key')
JWT_ALGO = 'HS256'
JWT_EXP_DELTA = timedelta(days=7)


# CORS config (for Flask-CORS)
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
CORS_RESOURCES = {r"/api/*": {"origins": [FRONTEND_URL]}}
CORS_SUPPORTS_CREDENTIALS = True
CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]
CORS_METHODS = ["GET", "POST", "OPTIONS"]


# JWT config for flask_jwt_extended
JWT_SECRET_KEY = JWT_SECRET
JWT_TOKEN_LOCATION = ["headers"]
JWT_HEADER_NAME = "Authorization"
JWT_HEADER_TYPE = "Bearer"
from datetime import timedelta
JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)
