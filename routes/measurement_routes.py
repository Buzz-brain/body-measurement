from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from controllers.measurement_controller import (
    get_user_measurements,
    create_measurement_record,
    get_measurement_record,
    get_image
)

measurement_bp = Blueprint('measurement', __name__, url_prefix='/api')

from flask_jwt_extended import jwt_required

@measurement_bp.route('/measurements', methods=['GET'])
@jwt_required()
def get_measurements_route():
    return get_user_measurements(request)

@measurement_bp.route('/measurements', methods=['POST'])
@jwt_required()
def create_measurement_route():
    try:
        return create_measurement_record(request)
    except Exception as e:
        return {'error': f'Failed to create measurement: {str(e)}'}, 500

from flask_jwt_extended import jwt_required

@measurement_bp.route('/measurements/<record_id>', methods=['GET'])
@jwt_required()
def get_measurement_record_route(record_id):
    return get_measurement_record(record_id)

@measurement_bp.route('/images/<image_id>', methods=['GET'])
def get_image_route(image_id):
    return get_image(image_id)
