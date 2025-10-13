import os
from datetime import timedelta

JWT_SECRET = os.environ.get('JWT_SECRET', 'dev_secret_key')
JWT_ALGO = 'HS256'
JWT_EXP_DELTA = timedelta(days=7)



# CORS config (for Flask-CORS)
# Explicitly list all allowed origins
CORS_RESOURCES = {
	r"/api/*": {
		"origins": [
			"http://localhost:5173",  # Vite dev server
			"http://localhost:3000",  # React default
			"https://virtual-body-measurement.vercel.app",  # Vercel frontend
			"https://aurore-chirurgic-lucy.ngrok-free.dev"  # ngrok tunnel
		]
	}
}
CORS_SUPPORTS_CREDENTIALS = True
CORS_ALLOW_HEADERS = ["Content-Type", "Authorization", "ngrok-skip-browser-warning"]
CORS_METHODS = ["GET", "POST", "OPTIONS"]


# JWT config for flask_jwt_extended
JWT_SECRET_KEY = JWT_SECRET
JWT_TOKEN_LOCATION = ["headers"]
JWT_HEADER_NAME = "Authorization"
JWT_HEADER_TYPE = "Bearer"
from datetime import timedelta
JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)
