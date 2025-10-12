from flask import Blueprint, request, jsonify
from controllers.auth_controller import signup, login
from controllers.auth_controller import logout

auth_bp = Blueprint('auth', __name__, url_prefix='/api')

@auth_bp.route('/signup', methods=['POST'])
def signup_route():
    return signup(request)

@auth_bp.route('/login', methods=['POST'])
def login_route():
    return login(request)

@auth_bp.route('/logout', methods=['POST'])
def logout_route():
    return logout(request)