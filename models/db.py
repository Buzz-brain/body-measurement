import os
from pymongo import MongoClient
import gridfs

# Only use environment variable for MongoDB URI
MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
	raise RuntimeError("MONGO_URI environment variable must be set for database connection.")
mongo_client = MongoClient(MONGO_URI)
default_db = mongo_client.get_default_database()
db = default_db if default_db is not None else mongo_client['bodymeasure']
fs = gridfs.GridFS(db)
measurements_coll = db['measurements']
USERS_COLL = db['users']
