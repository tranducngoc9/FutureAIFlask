# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import os, logging 
from os import path, getcwd
# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory, jsonify, json, Response, session
from flask_login         import login_user, logout_user, current_user
from flask_socketio   import SocketIO, emit, join_room, leave_room
from werkzeug.exceptions import HTTPException, NotFound, abort, Forbidden
from werkzeug.utils import secure_filename
from functools import wraps
import base64
import shutil, sys
import requests

# App modules
from app        import app, lm, db, bc, mail, socketio, q
from models import User, Role, Companies, Addresses, Cameras, Plans, Faces, Histories
from forms  import LoginForm, RegisterForm
from datatables import ColumnDT, DataTables
from datetime import date, timedelta, datetime
import base64
import time
from flask_mail import Message
import random
import string
from PIL import Image
from pprint import pprint
import numpy as np
import cv2
import sqlite3
from flask_jwt import JWT, jwt_required, current_identity
# from werkzeug.security import safe_str_cmp
from tasks import post_processing


basedir = os.path.abspath(os.path.dirname(__file__))
company_image_path = 'static/assets/img/company'
face_image_path = 'static/assets/img/face'
feature_db_path = 'static/db/'

def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)

@app.route('/api/launch', methods=['POST'])
def launch():
    output = json.dumps({"success": True})

    udid = request.form['udid']

    cam = db.session.query(Cameras).filter(Cameras.udid == udid).first()
    if  cam != None and cam.company_id:       
        if cam.company_id:
            db_path = os.path.join(basedir, feature_db_path + str(cam.company_id) + ".db")
            if not os.path.isfile(db_path):
                db_o_path = os.path.join(basedir, feature_db_path + "mq_feature_empty.db")
                shutil.copy2(db_o_path, db_path)

            company_face_dir = path.join(basedir, path.join(face_image_path + "/" + str(cam.company_id)))
            if not os.path.isdir(company_face_dir):
                os.mkdir(company_face_dir)

            return_output = json.dumps({"company_id": cam.company_id, "feature_path": "/" + feature_db_path + str(cam.company_id) + ".db"})   
            return success_handle(return_output)
        else:
             return error_handle("The camera is not assigned")
    else:
        print("Something happend")
        return error_handle("Cannot find the camera")
    
    return success_handle(output)


@app.route('/api/create_unknown', methods=['POST'])
def create_unknown():
    output = json.dumps({"success": True})

    udid = request.form['udid']
    filename = request.form['filename']
    feature = request.form['feature']
    #name = 'Unknown_' + time.strftime("%Y%m%d-%H%M%S")
    

    # print("Feature", feature);
    # print("Feature64", base64.b64decode(feature));
    print("Information of that face", filename)

    cam = db.session.query(Cameras).filter(Cameras.udid == udid).first()
    if  cam != None and cam.company_id:
        user = User(user=filename + "_" + str(cam.company_id), is_unknown=True, company_id=cam.company_id, full_name=filename + "_" + str(cam.company_id))
        user_role = db.session.query(Role).filter_by(name="user").first()
        user.roles = []
        user.roles.append(user_role)
        db.session.add(user)
        db.session.commit()

        if user:
            image = path.join(face_image_path + "/" + str(cam.company_id),  filename + ".jpg")
            face = Faces(user_id=user.id, user_id_o = user.id, file_name=image);
            db.session.add(face)
            db.session.commit()
            
            for i in range(20):
                tmp = path.join(face_image_path + "/" + str(cam.company_id), filename + "_" + str(i) + ".jpg")
                if os.path.isfile(path.join(basedir, tmp)):
                    face = Faces(user_id=user.id, user_id_o = user.id, file_name=tmp);
                    db.session.add(face)
            db.session.commit()
            if face:
                print("cool face has been saved")
                history = Histories(user_id=face.user_id, user_id_o = user.id, image=image, time=datetime.now(), camera=cam.id);
                db.session.add(history)
                db.session.commit()

                if history:
                    full_name = user.full_name
                    return_output = json.dumps({"id": history.id, "time": history.time, "user_name": full_name}) 

                    company = db.session.query(Companies).filter(Companies.id == cam.company_id).first()

                    if company:
                        data = {
                            "cam_id": cam.id,
                            "cam_ip": cam.ipaddr,
                            "image"  : image,
                            "time"       : datetime.now().timestamp(),
                            "user_data"   : user.to_dict()
                        }
                        job = q.enqueue_call(
                            func=post_processing, args=(company.secret, data, 4,), result_ttl=5000
                        )
                        print(job.get_id())

                    return success_handle(return_output)
                else:
                    print("An error saving history.")
                    return error_handle("An error saving history.")
            else:

                print("An error saving face image.")

                return error_handle("An error saving face image.")
        else:
             return error_handle("Cannot create a user")
    else:
        print("Something happend")
        return error_handle("Cannot find the camera")
    
    return success_handle(output)

@app.route('/api/image', methods=['POST'])
def image():
    output = json.dumps({"success": True})
    udid = request.form['udid']
    cam = db.session.query(Cameras).filter(Cameras.udid == udid).first()
    if  cam != None and cam.company_id:
        if 'file' not in request.files:

            print ("image is required")
            return error_handle("image is required.")
        else:

            
            file = request.files['file']
            print("File request type: ", file.mimetype)
            if file.mimetype not in app.config['file_allowed']:

                print("File extension is not allowed")

                return error_handle("We are only allow upload file with *.png , *.jpg")
            else:
                print("File is allowed and will be saved in ", face_image_path + "/" + str(cam.company_id))
                # get name in form data
                filename = request.form['filename']

                print("File name:", filename)
                img = cv2.imdecode(np.asarray(bytearray(file.read()), dtype="uint8"), cv2.IMREAD_COLOR)
                cv2.imwrite(path.join(basedir, path.join(face_image_path + "/" + str(cam.company_id), filename)), img)
                return_output = json.dumps({"file_path": path.join(face_image_path + "/" + str(cam.company_id), filename)})   
                return success_handle(return_output)
    
    else:
        print("Something happend")
        return error_handle("Cannot find the camera")

    return success_handle(output)

@app.route('/api/history', methods=['POST'])
def history():
    output = json.dumps({"success": True})

    # get name in form data
    udid = request.form['udid']
    filename = request.form['filename']
    image = request.form['image']
    feature = request.form['feature']

    # print("Feature", feature);
    # print("Feature64", base64.b64decode(feature));
    print("Information of that face", filename)
    cam = db.session.query(Cameras).filter(Cameras.udid == udid).first()
    if  cam != None and cam.company_id:
        face = db.session.query(Faces).filter(Faces.file_name == path.join(face_image_path + "/" + str(cam.company_id),  filename + ".jpg")).first()
        if  face != None:
            history = Histories(user_id=face.user_id, user_id_o=face.user_id, image=path.join(face_image_path + "/" + str(cam.company_id),  image), time=datetime.now(), camera=cam.id);
            db.session.add(history)
            db.session.commit()

            if history:
                user = User.query.filter(User.id == face.user_id).first()
                if user:

                    company = db.session.query(Companies).filter(Companies.id == cam.company_id).first()

                    if company:
                        data = {
                            "cam_id": cam.id,
                            "cam_ip": cam.ipaddr,
                            "image"  : path.join(face_image_path + "/" + str(cam.company_id),  image),
                            "time"       : datetime.now().timestamp(),
                            "user_data"   : user.to_dict()
                        }
                        job = q.enqueue_call(
                            func=post_processing, args=(company.secret, data, 5,), result_ttl=5000
                        )
                        print(job.get_id())


                    # send action to the camera
                    if user.has_roles("admin") or user.has_roles("staff"):
                        room = session.get(str(cam.company_id))
                        data = {}
                        data['camera_id'] = cam.udid
                        socketio.emit('action', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )


                    full_name = user.full_name
                    full_name = full_name.encode('utf8')
                    full_name = base64.b64encode(full_name)
                    return_output = json.dumps({"id": history.id, "time": history.time, "user_name": full_name.decode('utf-8')})   
                    return success_handle(return_output)
                else:
                    # Delete the face which has no user
                    db_path = feature_db_path + str(cam.company_id) + ".db"
                    if not os.path.isfile(db_path):
                        db_o_path = feature_db_path + "mq_feature_empty.db"
                        shutil.copy2(db_o_path, db_path)

                    conn = sqlite3.connect(db_path)
                    c = conn.cursor()

                    try:
                        sql_Delete_query = """DELETE FROM Features WHERE Features.name = '{0}'""".format(filename)
                        c.execute(sql_Delete_query)
                        Faces.query.filter_by(id=face.id).delete()
                    except sqlite3.IntegrityError as e:
                        print('Remove feature errror: ', e.args[0]) # column name is not unique
                    except Exception as e:
                        conn.rollback()
                        conn.close()
                        db.session.rollback()
                        return error_handle("An error delete face.")
                    #TODO: send socket to camera to remove
                    

                    room = session.get(str(cam.company_id))
                    data = {}
                    data['name'] = filename
                    socketio.emit('feature_del', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )

                    conn.commit()
                    conn.close()


                    return error_handle("Cannot find the user.")
            else:
                print("An error saving history.")

                return error_handle("An error saving history.")
        else:

            # Delete the no face
            db_path = feature_db_path + str(cam.company_id) + ".db"
            if not os.path.isfile(db_path):
                db_o_path = feature_db_path + "mq_feature_empty.db"
                shutil.copy2(db_o_path, db_path)

            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            try:
                sql_Delete_query = """DELETE FROM Features WHERE Features.name = '{0}'""".format(filename)
                print(sql_Delete_query)
                c.execute(sql_Delete_query)
            except sqlite3.IntegrityError as e:
                print('Remove feature errror: ', e.args[0]) # column name is not unique
            except Exception as e:
                conn.rollback()
                conn.close()
                return error_handle("An error delete face.")
            #TODO: send socket to camera to remove
            room = session.get(str(cam.company_id))
            data = {}
            data['name'] = filename
            socketio.emit('feature_del', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )

            conn.commit()
            conn.close()

            return error_handle("No face can found.")
    else:
        return error_handle("No cam can found.")
    
    return success_handle(output)


@app.route('/api/v1/add_webhook', methods=['POST'])
@jwt_required()
def add_webhook():
    #pprint(vars(current_identity))
    output = json.dumps({"success": True})
    link = request.json.get('link', None)
    
    if link:
        db.session.query(Companies).filter_by(id=current_identity.company_id).update({"secret": link}, synchronize_session='fetch')
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("No link error")


@app.route('/api/v1/add_account', methods=['POST'])
def add_account():
    #pprint(vars(current_identity))
    output = json.dumps({"success": True})
    
    name = request.form['name']
    phone = request.form['phone']
    mac = request.form['mac']

    print("========");
    print(name);
    print(phone);
    print(mac);
    return_output = json.dumps({"name": name, "phone":phone, "mac":mac}) 
    return success_handle(return_output)

@app.route('/api/v1/company')
@jwt_required()
def company():
    output = json.dumps({"success": True})
 
    company = db.session.query(Companies).filter(Companies.id == current_identity.company_id).first()
    if company:
        result = company.to_dict()
        return_output = json.dumps({"company": result}) 
        return success_handle(return_output)
    else:
        return error_handle("The company does not exist")

@app.route('/api/v1/cameras')
@jwt_required()
def cameras():
    output = json.dumps({"success": True})
    
    offset = request.args.get('offset') or 0
    limit = request.args.get('limit') or 20

    cameras = db.session.query(Cameras).filter(Cameras.company_id == current_identity.company_id).offset(offset).limit(limit).all()
    if cameras:
        data = []
        for c in cameras:
            temp = c.to_dict()
            data.append(temp)
        return_output = json.dumps({"cameras": data}) 
        return success_handle(return_output)
    else:
        return error_handle("Camera list is empty")

@app.route('/api/v1/addresses')
@jwt_required()
def addresses():
    output = json.dumps({"success": True})
    
    offset = request.args.get('offset') or 0
    limit = request.args.get('limit') or 20

    addresses = db.session.query(Addresses).filter(Addresses.company_id == current_identity.company_id).offset(offset).limit(limit).all()
    if addresses:
        data = []
        for a in addresses:
            temp = a.to_dict()
            data.append(temp)
        return_output = json.dumps({"addresses": data}) 
        return success_handle(return_output)
    else:
        return error_handle("Address list is empty")

@app.route('/api/v1/user/<user_id>')
@jwt_required()
def user(user_id):
    output = json.dumps({"success": True})
    
    user = db.session.query(User).filter(User.id == user_id).filter(User.company_id == current_identity.company_id).first()
    
    # job = q.enqueue_call(
    #     func=post_processing, args=(user.to_dict(), "0",), result_ttl=5000
    # )
    # print(job.get_id())

    if user:
        return_output = json.dumps({"user": user.to_dict()}) 
        return success_handle(return_output)
    else:
        return error_handle("User does not exist or you have no permission to access")



@app.route('/api/v1/user/<user_id>', methods=['POST'])
@jwt_required()
def user_update(user_id):
    output = json.dumps({"success": True})
    full_name = request.json.get('full_name', None)
    code = request.json.get('code', None)

    user = db.session.query(User).filter(User.id == user_id).filter(User.company_id == current_identity.company_id).first()

    if user:
        db.session.query(User).filter_by(id=user_id).update({"full_name": full_name}, synchronize_session='fetch')
        db.session.query(User).filter_by(id=user_id).update({"code": code}, synchronize_session='fetch')
        db.session.query(User).filter_by(id=user_id).update({"is_unknown": 0}, synchronize_session='fetch')
        db.session.commit()
        user = db.session.query(User).filter(User.id == user_id).filter(User.company_id == current_identity.company_id).first()
        return_output = json.dumps({"user": user.to_dict()}) 
        return success_handle(return_output)
    else:
        return error_handle("User does not exist or you have no permission to access")


@app.route('/api/v1/address/<address_id>')
@jwt_required()
def address(address_id):
    output = json.dumps({"success": True})
    
    address_ = db.session.query(Addresses).filter(Addresses.id == address_id).filter(Addresses.company_id == current_identity.company_id).first()
    if address_:
        return_output = json.dumps({"address": address_.to_dict()}) 
        return success_handle(return_output)
    else:
        return error_handle("Address does not exist or you have no permission to access")

@app.route('/api/v1/camera/<camera_id>')
@jwt_required()
def camera(camera_id):
    output = json.dumps({"success": True})
    
    camera = db.session.query(Cameras).filter(Cameras.id == camera_id).filter(Cameras.company_id == current_identity.company_id).first()
    if camera:
        return_output = json.dumps({"camera": camera.to_dict()}) 
        return success_handle(return_output)
    else:
        return error_handle("Camera does not exist or you have no permission to access")


@app.route('/api/v1/users')
@jwt_required()
def user_list():
    output = json.dumps({"success": True})
    
    offset = request.args.get('offset') or 0
    limit = request.args.get('limit') or 20


    users = db.session.query(User).filter(User.company_id == current_identity.company_id).offset(offset).limit(limit).all()
    if users:
        data = []
        for u in users:
            temp = u.to_dict()
            data.append(temp)
        return_output = json.dumps({"users": data}) 
        
        return success_handle(return_output)
    else:
        return error_handle("User list is empty")

@app.route('/api/v1/histories')
@jwt_required()
def histories():
    output = json.dumps({"success": True})
    
    offset = request.args.get('offset') or 0
    limit = request.args.get('limit') or 20
    group_by = request.args.get('group_by') or 0
    unknown = request.args.get('unknown') or -1
    camera_id = request.args.get('camera_id') or -1

    group_by = int(group_by)
    unknown = int(unknown)
    camera_id = int(camera_id)

    query = db.session.query(Histories).join(Cameras).join(User).filter(User.company_id == current_identity.company_id)
    if camera_id > 0:
        query = query.filter(Cameras.company_id == current_identity.company_id).filter(Cameras.id == camera_id)
    else:
        query = query.filter(Cameras.company_id == current_identity.company_id)

    if unknown > 0:
        query = query.filter(User.is_unknown == unknown)

    if group_by:
        query = query.group_by(Histories.user_id)

    query = query.order_by(Histories.time.desc())
 
    histories = query.offset(offset).limit(limit).all()

    if histories:
        data = []
        for h in histories:
            temp = h.to_dict()
            data.append(temp)
        return_output = json.dumps({"histories": data}) 
        return success_handle(return_output)
    else:
        return error_handle("History list is empty")


@app.route('/api/v1/histories_detail')
@jwt_required()
def histories_detail():
    output = json.dumps({"success": True})
    
    offset = request.args.get('offset') or 0
    limit = request.args.get('limit') or 20
    group_by = request.args.get('group_by') or 0
    unknown = request.args.get('unknown') or -1
    camera_id = request.args.get('camera_id') or -1

    group_by = int(group_by)
    unknown = int(unknown)
    camera_id = int(camera_id)

    query = db.session.query(Histories).join(Cameras).join(User).filter(User.company_id == current_identity.company_id)
    if camera_id > 0:
        query = query.filter(Cameras.company_id == current_identity.company_id).filter(Cameras.id == camera_id)
    else:
        query = query.filter(Cameras.company_id == current_identity.company_id)

    if unknown > 0:
        query = query.filter(User.is_unknown == unknown)

    if group_by:
        query = query.group_by(Histories.user_id)

    query = query.order_by(Histories.time.desc())

    histories = query.offset(offset).limit(limit).all()

    if histories:
        data = []
        for h in histories:
            temp = h.to_dict()
            
            if (temp["camera"]):
                camera = db.session.query(Cameras).filter(Cameras.id == temp["camera"]).first()
                if camera:
                     temp["camera"] = camera.to_dict()
            user = db.session.query(User).filter(User.id == temp["user_id"]).first()

            if user:
                temp["user"] = user.to_dict()
                del temp['user_id']
            data.append(temp)
        return_output = json.dumps({"histories_detail": data}) 
        return success_handle(return_output)
    else:
        return error_handle("History list is empty")

@app.route('/api/test', methods=['POST'])
def test():
    output = json.dumps({"success": True})
    
    event = request.json.get('event', None)
    if (event):
        print(event);
        data = request.json.get('data', None)

        if data:
            print(data);
            return_output = json.dumps({"data": data}) 
            return success_handle(return_output)
        return success_handle(output)
    else:
        return error_handle("Invalid request")
