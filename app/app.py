#!/usr/bin/env python
import os
import json as js
import sqlite3
import os.path
import shutil, sys
from datetime import timedelta

from flask            import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask_login      import LoginManager
from flask_bcrypt     import Bcrypt
from flask_mail 	  import Mail
from flask_cors 	  import CORS
from flask_socketio   import SocketIO, emit, join_room, leave_room
from flask_babel import Babel
from flask_migrate import Migrate
from flask_jwt import JWT
from flask_bcrypt import Bcrypt

from rq import Queue
from rq.job import Job
from worker import conn
from flask_whooshee import Whooshee
from flask_dropzone import Dropzone

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))
feature_db_path = os.path.join(basedir, 'static/db/')


app = Flask(__name__)

app.config.from_object('configuration.Config')
app.config['file_allowed'] = ['image/png', 'image/jpeg', 'application/octet-stream']
app.config['JWT_EXPIRATION_DELTA'] = timedelta(days=365)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


#mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root ./mlruns
#Dropzone
#https://flask-dropzone.readthedocs.io/en/latest/configuration.html#file-type-filter

mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": 'mq.dev1@mqsolutions.com.vn',
    "MAIL_PASSWORD": '123456789'
}

app.config.update(mail_settings)
db = SQLAlchemy(app) # flask-sqlalchemy
bc = Bcrypt(app) # flask-bcrypt
mail = Mail(app)
lm = LoginManager(   ) # flask-loginmanager
lm.init_app(app) # init the login manager
babel = Babel(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
q = Queue(connection=conn)
whooshee = Whooshee(app)
dropzone = Dropzone(app)

socketio = SocketIO(app)


CORS(app, supports_credentials=True, allow_headers=['Content-Type', 'X-ACCESS_TOKEN', 'Authorization'])

# Setup database
@app.before_first_request
def initialize_database():
    db.create_all()

# Import routing, models and Start the App
from views import *
from models import *
from api import *

def authenticate(username, password):
    user = User.query.filter(User.user == username).first()

    if user:
        if (user.has_roles("user") or user.has_roles("superuser")):
            if user.company_id and bcrypt.check_password_hash(user.password, password):
                return user

def identify(payload):
    return User.query.filter(User.id == payload['identity']).scalar()


jwt = JWT(app, authenticate, identify)



def insertFeatureToDB(name, feature, company_id):
    #print("insertFeatureToDB")
    db_path = feature_db_path + company_id + ".db"
    if not os.path.isfile(db_path):
    	db_o_path = feature_db_path + "mq_feature_empty.db"
    	shutil.copy2(db_o_path, db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
      c.execute('''INSERT INTO Features (name, data) VALUES (?, ?)''', (name, feature,))
    except sqlite3.IntegrityError as e:
      print('Insert feature errror: ', e.args[0]) # column name is not unique
    conn.commit()

    conn.close()


# When the client emits 'connection', this listens and executes
@socketio.on('status', namespace='/camera')
def camera_connected(data):
    print ('Camera connected:', data)
    status = js.loads(data)
    if not 'version' in status:
        status['version'] = 0
    cam = db.session.query(Cameras).filter(Cameras.udid == status['udid']).first()
    if cam != None:
        db.session.query(Cameras).filter(Cameras.udid == status['udid']).update({"ipaddr": status['ip_address'], "time": datetime.now(), "version": status['version']}, synchronize_session='fetch')
    else:
        cam = Cameras(udid=status['udid'], version= status['version'], ipaddr=status['ip_address'], time=datetime.now());
        db.session.add(cam)

    db.session.commit()
    print("===============camera joined: " + status['ip_address'])
    if cam.company_id:
        room = session.get(str(cam.company_id))
        join_room(room)
    else:
        room = session.get("camera_global")
        join_room(room)

# When the client emits 'new message', this listens and executes
@socketio.on('new face', namespace='/camera')
def new_face(data):
    #print ('New face:', data)
    feature_data = js.loads(data)
    if feature_data and feature_data['udid']:
        cam = db.session.query(Cameras).filter(Cameras.udid == feature_data['udid']).first()
        if cam and cam.company_id:
            insertFeatureToDB(feature_data['name'], feature_data['feature'], str(cam.company_id))
            room = session.get(str(cam.company_id))
            socketio.emit('feature', { 'data' :  data }, namespace='/camera', room=room )

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=3456, debug=True)
    # whooshee.reindex()
    socketio.run(app, host='0.0.0.0', port=3456, debug=True)