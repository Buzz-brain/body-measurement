# Vercel serverless entrypoint for the Flask app
# Vercel's Python builder will use this file to serve your WSGI `app` object.

from app import app  # imports the Flask `app` defined in project root `app.py`

# Expose `app` variable — the Vercel Python runtime will use this WSGI callable.

