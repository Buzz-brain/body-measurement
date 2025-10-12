
from flask import request, jsonify
from datetime import datetime, timedelta
import bcrypt
import os
from models.db import db, USERS_COLL
from config import JWT_SECRET, JWT_ALGO, JWT_EXP_DELTA
from flask_jwt_extended import create_access_token


# Deprecated: create_jwt. Use flask_jwt_extended's create_access_token instead.

JWT_BLACKLIST = set()

def signup(request):
    data = request.json
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    if USERS_COLL.find_one({'email': email}):
        return jsonify({'error': 'Email already registered'}), 409
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user = {
        'email': email,
        'password_hash': pw_hash,
        'name': data.get('name', ''),
        'created_at': datetime.utcnow()
    }
    result = USERS_COLL.insert_one(user)
    user_id = str(result.inserted_id)
    user_obj = {
        'id': user_id,
        'email': email,
        'name': user.get('name', ''),
        'createdAt': user['created_at'].isoformat() + 'Z'
    }
    from datetime import timedelta
    token = create_access_token(
        identity=user_id,
        additional_claims={
            "email": email,
            "name": user.get('name', ''),
            "created_at": user['created_at'].isoformat() + 'Z'
        },
        expires_delta=timedelta(days=7)
    )
    return jsonify({'token': token, 'user': user_obj}), 201

def login(request):
    data = request.json
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    user = USERS_COLL.find_one({'email': email})
    if not user or not bcrypt.checkpw(password.encode(), user['password_hash']):
        return jsonify({'error': 'Invalid email or password'}), 401
    user_id = str(user['_id'])
    user_obj = {
        'id': user_id,
        'email': user['email'],
        'name': user.get('name', ''),
        'createdAt': user.get('created_at', datetime.utcnow()).isoformat() + 'Z'
    }
    from datetime import timedelta
    token = create_access_token(
        identity=user_id,
        additional_claims={
            "email": user['email'],
            "name": user.get('name', ''),
            "created_at": user_obj['createdAt']
        },
        expires_delta=timedelta(days=7)
    )
    return jsonify({'token': token, 'user': user_obj}), 200

def logout(request):
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid Authorization header'}), 401
    token = auth_header.split(' ', 1)[1]
    JWT_BLACKLIST.add(token)
    return jsonify({'message': 'Logged out successfully'}), 200