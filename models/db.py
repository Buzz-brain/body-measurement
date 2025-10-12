import os
from pymongo import MongoClient
import gridfs

default_mongo_uri = "mongodb+srv://chinomsochristian03_db_user:VPRrbvrnJAAiyC3v@cluster0.tj4qti1.mongodb.net/bodymeasure?retryWrites=true&w=majority&appName=Cluster0"
MONGO_URI = os.environ.get('MONGO_URI', default_mongo_uri)
mongo_client = MongoClient(MONGO_URI)
default_db = mongo_client.get_default_database()
db = default_db if default_db is not None else mongo_client['bodymeasure']
fs = gridfs.GridFS(db)
measurements_coll = db['measurements']
USERS_COLL = db['users']
