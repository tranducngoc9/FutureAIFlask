# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import os, logging 
from os import path, getcwd
import sqlite3
import shutil, sys

# Flask modules
from flask               import render_template, request, url_for, redirect, send_from_directory, jsonify, json, Response, session
from flask_login         import login_user, logout_user, current_user
from flask_socketio   import SocketIO, emit, join_room, leave_room
from werkzeug.exceptions import HTTPException, NotFound, abort, Forbidden
from werkzeug.utils import secure_filename
from functools import wraps
import base64
from sqlalchemy import func, text
from sqlalchemy.sql import label
from sqlalchemy.exc import SQLAlchemyError
from configuration import Config
from flask_babel import Babel
from vnpay import vnpay

# App modules
from app        import app, lm, db, bc, mail, socketio, babel, whooshee
from models import User, Role, Companies, Addresses, Cameras, Plans, Faces, Histories, Versions, Dataset
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

import face_preprocess
import numpy as np
import cv2
import mxnet as mx
import sklearn
from sklearn.decomposition import PCA
import zipfile
import hub
import pandas as pd
from pandas_profiling import ProfileReport
import chardet
#from mtcnn.mtcnn import MTCNN
#import faiss


#detector = MTCNN()
basedir = os.path.abspath(os.path.dirname(__file__))
company_image_path = 'static/assets/img/company'
face_image_path = 'static/assets/img/face'
feature_db_path = os.path.join(basedir, 'static/db/')
static_path = 'static'
version_path = 'static/assets/version'
DATASET_INDEX = 'index.bin'
DATASET_LABELS = 'labels.pkl'
DATASET_DIR = os.path.join(static_path, 'dataset')

#index = faiss.read_index(os.path.join(DATASET_PATH, DATASET_INDEX)) # load index

ctx = mx.cpu(0)
VERSION_ALLOWED_EXTENSIONS = set(['zip', 'tar'])

image_size = (112,112)
model_path = "models/model-y1-test2/model,0"
ga_model_path = "models/gender-age/model,0"

def get_ga(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    ret = model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))
    return gender, age

def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model

def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)

def is_same_company(company_id):
    if (current_user.has_roles("superuser")):
        return True

    if (current_user.company_id == company_id):
        return True

    return False

def get_user_path():
    user_id = current_user.id
    user_path = os.path.join(DATASET_DIR, str(user_id))
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    return user_path

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_active or not current_user.is_authenticated:
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function

def user_is(role):
    """
    Takes an role (a string name of either a role or an ability) and returns the function if the user has that role
    """
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if (current_user.has_roles("superuser")):
                return func(*args, **kwargs)

            if role in [r.name for r in current_user.roles]:
                if current_user.company_id:
                    return func(*args, **kwargs)
            raise Forbidden("You do not have access")
        return inner
    return wrapper


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in VERSION_ALLOWED_EXTENSIONS


@app.cli.command()
def initdb():
    """
    Populate a small db with some example entries.
    """
    print("build_sample_db")

    db.drop_all()
    db.create_all()

    with app.app_context():
        user_role = Role(name='user')
        super_user_role = Role(name='superuser')
        db.session.add(user_role)
        db.session.add(super_user_role)
        db.session.commit()

        admin_user = User(
            user='admin',
            email='ams@mqsolutions.vn',
            password=bc.generate_password_hash('MQ1234')
            
        )
        admin_user.roles.append(super_user_role)
        cadmin_user = User(
            user='cadmin',
            email='admin1@gmail.com',
            password=bc.generate_password_hash('MQ1234'),
            company_id = 1
        )
        cadmin_user.roles.append(user_role)
       
        db.session.add(admin_user)
        db.session.add(cadmin_user)

        db.session.commit()

        plan1 = Plans(name=u"Free");
        plan2 = Plans(name=u"Standard");
        plan3 = Plans(name=u"Advance");
        db.session.add(plan1)
        db.session.add(plan2)
        db.session.add(plan3)
        db.session.commit()

        company1 = Companies(name="Đơn vị 1", email="a1@gmail.com", phone="123456789", address="address", plan_id=plan1.id);
        company2 = Companies(name="Đơn vị 2", email="a1@gmail.com", phone="123456789", address="address", plan_id=plan2.id);
        company1.plan_id = plan1.id
        company2.plan_id = plan2.id
        db.session.add(company1)
        db.session.add(company2)
        db.session.commit()

        address1 = Addresses(name="address 1", address=" 12 Khuat duy tien");
        address2 = Addresses(name="address 2", address=" 12 Khuat duy tien 2");
        db.session.add(address1)
        db.session.add(address2)
        db.session.commit()

    print("build_sample_db done")
    return

# provide login manager with load_user callback
@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Logout user
@app.route('/logout.html')
def logout():
    logout_user()
    return redirect(url_for('index'))

# Register a new user
@app.route('/register.html', methods=['GET', 'POST'])
def register():
    
    # declare the Registration Form
    form = RegisterForm(request.form)

    msg = None

    if request.method == 'GET': 

        return render_template( 'pages/register.html', form=form, msg=msg )

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str) 
        email    = request.form.get('email'   , '', type=str) 

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = User.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'
        
        else:         

            pw_hash = bc.generate_password_hash(password)

            user = User(username, email, pw_hash)

            user.save()

            msg = 'User created, please <a href="' + url_for('login') + '">login</a>'     

    else:
        msg = 'Input error'     

    return render_template( 'pages/register.html', form=form, msg=msg )

# Authenticate user
@app.route('/login.html', methods=['GET', 'POST'])
def login():
    
    # Declare the login form
    form = LoginForm(request.form)
    data = (request.form).to_dict(flat=False)
    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if data:

        # assign form data to variables
        username = request.form.get('user', '', type=str)
        print(username)
        password = request.form.get('pass', '', type=str)
        print(password)

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()
        if user:
            
            if user.password and bc.check_password_hash(user.password, password):
                login_user(user)
                print("=========>>>>>")
                pprint(vars(current_user))
                return redirect(url_for('index'))
            else:
                msg = u"Mật khẩu sai! hãy thử lại."
        else:
            msg = u"Người dùng không tồn tại"

    return render_template( 'pages/login.html', form=form, msg=msg )





#===========================

@app.route('/detail-timeline.html')
@login_required
@user_is("user")
def detailtimeline():

    user_id = request.args.get('id')
    if not user_id:
        user_id = current_user.id

    users = db.session.query(User).join(User.roles).filter(User.id == user_id).filter(User.is_unknown == False).filter(Role.name == "staff").first()

    

    #print(histories)
    return render_template( 'pages/detail-timeline.html', users=users)


@app.route('/detail_time_dashboard', methods=['POST'])
@login_required
@user_is("user")
def detail_time_dashboard():
    output = json.dumps({"success": True})
    user_id = request.form['user_id']
    startDate = request.form['startDate']
    endDate = request.form['endDate'] + ' 23:59:59'
    startDatetime = date.today()
    endDatetime = date.today()
    if startDate and endDate:
        startDatetime = datetime.strptime(startDate,"%Y-%m-%d").date()
        endDatetime = datetime.strptime(endDate,"%Y-%m-%d %H:%M:%S")

        in_late_count = 0
        out_early_count = 0
        escape_count = 0

        histories = db.session.query(Cameras, Histories.id, func.min(Histories.time), func.max(Histories.time)).join(Cameras, Histories.camera == Cameras.id).filter(Histories.user_id == user_id).filter(Histories.time <= endDatetime).filter(Histories.time >= startDatetime).group_by(func.strftime("%Y-%m-%d", Histories.time)).all()
        
        for cam, history_id, history_start, history_end in histories:
            address = db.session.query(Addresses).join(Cameras, Cameras.address_id == Addresses.id).filter(Cameras.id == cam.id).first()
            if address and history_start and address.start and address.start < history_start.time():
                in_late_count = in_late_count + 1
           
            if address and history_end and address.end and address.end > history_end.time():
                out_early_count = out_early_count + 1

        escape_count = len(histories)
        return_output = json.dumps({"escape_count": escape_count, "in_late_count": in_late_count, "out_early_count": out_early_count})  
        return success_handle(return_output)
    else:
        return error_handle("date is empty.")


@app.route('/time_dashboard', methods=['POST'])
@login_required
@user_is("user")
def time_dashboard():
    output = json.dumps({"success": True})
    selected_date =  request.form['selected_date']

    if selected_date:
        selected_datetime = datetime.strptime(selected_date,"%Y-%m-%d").date()
        print("==================")
        print(selected_datetime)
        print("==================")
        users = db.session.query(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(User.is_unknown == False).filter(Role.name == "staff").all()

        in_late_count = 0
        out_early_count = 0
        escape_count = 0
        for user in users:
            address_start, h_start, start = db.session.query(Addresses, Histories, func.min(Histories.time)).join(Cameras, Cameras.id==Histories.camera).join(Addresses, Cameras.address_id==Addresses.id).filter(Histories.user_id == user.id).filter(Histories.time <= selected_datetime + timedelta(days=1)).filter(Histories.time >= selected_datetime).first()
            address_end, h_end, end = db.session.query(Addresses, Histories, func.max(Histories.time)).join(Cameras, Cameras.id==Histories.camera).join(Addresses, Cameras.address_id==Addresses.id).filter(Histories.user_id == user.id).filter(Histories.time <= selected_datetime + timedelta(days=1)).filter(Histories.time >= selected_datetime).first()
            if address_start and start and address_start.start and address_start.start < start.time():
                in_late_count = in_late_count + 1
           
            if address_start and end and address_start.end and address_start.end > end.time():
                out_early_count = out_early_count + 1

            if not start:
                escape_count = escape_count + 1
        return_output = json.dumps({"escape_count": escape_count, "in_late_count": in_late_count, "out_early_count": out_early_count})  
        return success_handle(return_output)
    else:
        return error_handle("date is empty.")


@app.route('/time_data')
@login_required
@user_is("user")
def time_data():

    selected_date = request.args.get('selected_date')
    selected_datetime = date.today()
    if selected_date:
        selected_datetime = datetime.strptime(selected_date,"%Y-%m-%d").date()
        print("==================")
        print(selected_datetime)
        print("==================")
    # defining columns
    columns = [
        ColumnDT(User.id),
        ColumnDT(User.user),
        ColumnDT(User.full_name),
        ColumnDT(User.position),
        ColumnDT(User.id),
        ColumnDT(User.id),
        ColumnDT(User.id),
        ColumnDT(User.id),
        ColumnDT(User.id),
    ]
 
    # defining the initial query depending on your purpose
    query = db.session.query().select_from(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(User.is_unknown == False).filter(Role.name == "staff")

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    for i in range(len(rowTable.output_result()["data"])):
        address_start, h_start, start = db.session.query(Addresses, Histories, func.min(Histories.time)).join(Cameras, Cameras.id==Histories.camera).join(Addresses, Cameras.address_id==Addresses.id).filter(Histories.user_id == rowTable.output_result()["data"][i]['0']).filter(Histories.time <= selected_datetime + timedelta(days=1)).filter(Histories.time >= selected_datetime).first()
        address_end, h_end, end = db.session.query(Addresses,Histories, func.max(Histories.time)).join(Cameras, Cameras.id==Histories.camera).join(Addresses, Cameras.address_id==Addresses.id).filter(Histories.user_id == rowTable.output_result()["data"][i]['0']).filter(Histories.time <= selected_datetime + timedelta(days=1)).filter(Histories.time >= selected_datetime).first()
        if start:
            rowTable.output_result()["data"][i]['4'] = start.strftime("%I:%M %p")
        else:
            rowTable.output_result()["data"][i]['4'] = ''

        if end:
            rowTable.output_result()["data"][i]['5'] = end.strftime("%I:%M %p")
        else:
            rowTable.output_result()["data"][i]['5'] = ''

        if start and end:
            elapsedTime = end - start
            hours = divmod(elapsedTime.total_seconds(), 3600)
            minutes = divmod(hours[1], 60)
            rowTable.output_result()["data"][i]['6'] = '%d giờ, %d phút' % (hours[0],minutes[0])
            pprint(rowTable.output_result()["data"][i]['6'])
        else:
            rowTable.output_result()["data"][i]['6'] = ''

        rowTable.output_result()["data"][i]['7'] = 0
        rowTable.output_result()["data"][i]['8'] = 0
        if (address_start and start and address_start.start):
            rowTable.output_result()["data"][i]['7'] = address_start.start < start.time()
        if (address_end and end and address_end.end):
            rowTable.output_result()["data"][i]['8'] = address_end.end > end.time()

    #print(rowTable.output_result())

    return jsonify(rowTable.output_result())



@app.route('/detail_time_data')
@login_required
@user_is("user")
def detail_time_data():
    user_id = request.args.get('user_id')
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate') + ' 23:59:59'
    startDatetime = date.today()
    endDatetime = date.today()
    if startDate and endDate:
        startDatetime = datetime.strptime(startDate,"%Y-%m-%d").date()
        endDatetime = datetime.strptime(endDate,"%Y-%m-%d %H:%M:%S")
        print("==================")
        print(startDatetime)
        print(endDatetime)
        print("==================")

    # defining columns
    columns = [
        ColumnDT(Histories.id),
        ColumnDT(func.strftime("%Y-%m-%d", Histories.time)),
        ColumnDT(func.count(Histories.id)),
        ColumnDT(func.min(Histories.time)),
        ColumnDT(func.max(Histories.time)),
        ColumnDT(Cameras.id),
        ColumnDT(Cameras.id),
        ColumnDT(Cameras.id),
        ColumnDT(Cameras.id),
    ]
 
    # defining the initial query depending on your purpose
    #query = db.session.query().select_from(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(User.is_unknown == False).filter(Role.name == "staff")
    query = db.session.query().select_from(Histories, Cameras).join(Cameras, Cameras.id == Histories.camera).filter(Histories.user_id == user_id).filter(Histories.time <= endDatetime).filter(Histories.time >= startDatetime).group_by(func.strftime("%Y-%m-%d", Histories.time))
    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    for i in range(len(rowTable.output_result()["data"])):
        if rowTable.output_result()["data"][i]['3'] and rowTable.output_result()["data"][i]['4']:
            elapsedTime = rowTable.output_result()["data"][i]['4'] - rowTable.output_result()["data"][i]['3']
            hours = divmod(elapsedTime.total_seconds(), 3600)
            minutes = divmod(hours[1], 60)
            rowTable.output_result()["data"][i]['5'] = '%d giờ, %d phút' % (hours[0],minutes[0])

        cam_id = int(rowTable.output_result()["data"][i]['6'])
        address = db.session.query(Addresses).join(Cameras, Cameras.address_id == Addresses.id).filter(Cameras.id == cam_id).first()
        rowTable.output_result()["data"][i]['6'] = 0
        rowTable.output_result()["data"][i]['7'] = 0
        if (address and address.start and address.end):
            rowTable.output_result()["data"][i]['6'] = address.start < rowTable.output_result()["data"][i]['3'].time()
            rowTable.output_result()["data"][i]['7'] = address.end > rowTable.output_result()["data"][i]['4'].time()

        if rowTable.output_result()["data"][i]['3']:
            rowTable.output_result()["data"][i]['3'] = rowTable.output_result()["data"][i]['3'].strftime("%I:%M %p")

        if rowTable.output_result()["data"][i]['4']:
            rowTable.output_result()["data"][i]['4'] = rowTable.output_result()["data"][i]['4'].strftime("%I:%M %p")




    print(rowTable.output_result())

    return jsonify(rowTable.output_result())


@app.route('/version_data')
@login_required
@user_is("superuser")
def version_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(Versions.id),
        ColumnDT(Versions.name),
        ColumnDT(Versions.file),
        ColumnDT(func.strftime("%Y-%m-%d %H:%M:%S", Versions.confirmed_at))
    ]
 
    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Versions)

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    #print(rowTable.output_result())

    return jsonify(rowTable.output_result())

@app.route('/del_version', methods=['POST'])
@login_required
@user_is("superuser")
def del_version():
    output = json.dumps({"success": True})
    version_id =  request.form['id']

    ret = Versions.query.filter_by(id=version_id).delete()
    if ret:
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete camera.")

@app.route('/manager_version_camera.html')
@login_required
@user_is("superuser")
def manager_version_camera():
    return render_template( 'pages/manager_version_camera.html')

@app.route('/add_version', methods=['POST'])
@login_required
@user_is("superuser")
def add_version():
    output = json.dumps({"success": True})
    filename = request.form['name']
    file_path = ""
    if filename:
        file = request.files['inputFile']
        if file and allowed_file(file.filename):
            urlSafeEncodedStr =file.filename
            file_path = path.join(version_path, urlSafeEncodedStr);
            file.save(path.join(basedir, file_path))
            version = Versions(name=filename, file=file_path)
            db.session.add(version)
            db.session.commit()
            if version:
                return redirect("manager_version_camera.html")
            else:
                print("An error saving version.")
                return error_handle("An error saving version.")
        else:
            return error_handle("File not allowed.")
    else:
        return error_handle("file is empty.")

#===========================

@app.route('/history_data_list', methods=['GET'])
@login_required
@user_is("user")
def history_data_list():
    """Return server side data."""
    # defining columns
    page = int(request.args.get('page'))
    size = int(request.args.get('size'))
    type = int(request.args.get('type'))
    if type == 0:
        histories = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).group_by(Histories.time).order_by(Histories.time.desc()).offset(page*size).limit(size).all()
    elif type == 1:
        histories = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).filter(User.is_unknown == False).group_by(Histories.time).order_by(Histories.time.desc()).offset(page*size).limit(size).all()
    elif type == 2:
        histories = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).filter(User.is_unknown == True).group_by(Histories.time).order_by(Histories.time.desc()).offset(page*size).limit(size).all()
    else:
        histories = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).group_by(Histories.time).order_by(Histories.time.desc()).offset(page*size).limit(size).all()

    return render_template('pages/history_list.html', histories=histories)


# @app.route('/history.html')
# @login_required
# @user_is("user")
# def history_data():
#     histories = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).group_by(Histories.time).order_by(Histories.time.desc()).all()
#     histories_known = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).filter(User.is_unknown == False).group_by(Histories.time).order_by(Histories.time.desc()).all()
#     histories_unknown = db.session.query(Histories, Cameras, User, Faces).join(Cameras).join(User).outerjoin(Faces).filter(Cameras.company_id == current_user.company_id).filter(User.company_id == current_user.company_id).filter(User.is_unknown == True).group_by(Histories.time).order_by(Histories.time.desc()).all()
#     pprint(histories)
#     return render_template( 'pages/history.html', histories=histories, histories_known=histories_known, histories_unknown=histories_unknown)

#===========================

@app.route('/managercompany.html')
@login_required
@user_is("superuser")
def managercompany():
    plans = db.session.query(Plans).all()
    return render_template( 'pages/managercompany.html', plans=plans)


@app.route('/companies_data')
@login_required
@user_is("superuser")
def companies_data():
    """Return server side data."""
    # defining columns

    columns = [
        ColumnDT(Companies.id),
        ColumnDT(Companies.name),
        ColumnDT(Plans.name),
        ColumnDT(Companies.email),
        ColumnDT(Companies.phone),
        ColumnDT(Companies.address),
        ColumnDT(Companies.secret)
    ]

    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Companies).outerjoin(Plans).filter()

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    # print(rowTable.output_result())
    return jsonify(rowTable.output_result())


@app.route('/add_company', methods=['POST'])
@login_required
@user_is("superuser")
def add_company():
    output = json.dumps({"success": True})
    addName =  request.form['addName']
    addEmail =  request.form['addEmail']
    addPhone =  request.form['addPhone']
    addPlan =  request.form['addPlan']
    addAddress =  request.form['addAddress']
    file = None
    file_image_path = ""

    if addName:
        if 'file' in request.files:
            file = request.files['file']

        if file:
            if file.mimetype not in ['image/png', 'image/jpeg', 'application/octet-stream']:
                print("File extension is not allowed")
                return error_handle("We are only allow upload file with *.png , *.jpg")
            else:
                name = str(int(time.time())) + secure_filename(file.filename)
                urlSafeEncodedBytes = base64.urlsafe_b64encode(name.encode("utf-8"))[:21]
                urlSafeEncodedStr = str(urlSafeEncodedBytes) + ".jpg"

                file_image_path = path.join(company_image_path, urlSafeEncodedStr);
                file.save(path.join(basedir, file_image_path))
        company = Companies(name= addName, email=addEmail, phone=addPhone, address=addAddress, plan_id = addPlan, logo_image = file_image_path, secret = randomString(20))
        db.session.add(company)
        db.session.commit()
        if company:
            company_face_dir = path.join(basedir, path.join(face_image_path + "/" + str(company.id)))
            if not os.path.isdir(company_face_dir):
                os.mkdir(company_face_dir)

            return success_handle(output)
        else:
            print("An error saving company.")
            return error_handle("An error saving company.")
    else:
        return error_handle("company name is empty.")

@app.route('/del_company', methods=['POST'])
@login_required
@user_is("superuser")
def del_company():
    output = json.dumps({"success": True})
    company_id =  request.form['id']

    ret = Companies.query.filter_by(id=company_id).delete()
    if ret:
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete camera.")

@app.route('/detail-company.html')
@login_required
@user_is("superuser")
def detailcompany():
    plans = db.session.query(Plans).all()
    company_id = request.args.get('id')
    company = db.session.query(Companies).filter(Companies.id == company_id).outerjoin(Plans).first()
    if company:
        return render_template( 'pages/detail-company.html', company=company, plans = plans)
    else:
        return redirect(url_for('managercamera'))


@app.route('/edit_company', methods=['POST'])
@login_required
@user_is("superuser")
def edit_company():
    output = json.dumps({"success": True})
    company_id =  request.form['id']
    name =  request.form['name']
    email =  request.form['email']
    phone =  request.form['phone']
    plan =  request.form['plan']
    address =  request.form['address']
    secret =  request.form['secret']
    file = None
    file_image_path = ""
    if company_id:
        if 'file' in request.files:
            file = request.files['file']

        if file:
            if file.mimetype not in ['image/png', 'image/jpeg', 'application/octet-stream']:
                print("File extension is not allowed")
                return error_handle("We are only allow upload file with *.png , *.jpg")
            else:
                filename = str(int(time.time())) + secure_filename(file.filename)
                urlSafeEncodedBytes = base64.urlsafe_b64encode(name.encode("utf-8"))[:21]
                urlSafeEncodedStr = str(urlSafeEncodedBytes) + ".jpg"

                file_image_path = path.join(company_image_path, urlSafeEncodedStr);
                file.save(path.join(basedir, file_image_path))
        company = Companies.query.filter_by(id=company_id)
        if company:
            if file_image_path:
                db.session.query(Companies).filter_by(id=company_id).update({"name": name, "email": email, "phone": phone, "plan_id": plan, "secret": secret, "address": address, "logo_image": file_image_path}, synchronize_session='fetch')
            else:
                db.session.query(Companies).filter_by(id=company_id).update({"name": name, "email": email, "phone": phone, "plan_id": plan, "secret": secret, "address": address}, synchronize_session='fetch')
            db.session.commit()

            company_face_dir = path.join(basedir, path.join(face_image_path + "/" + str(company_id)))
            if not os.path.isdir(company_face_dir):
                os.mkdir(company_face_dir)

            return success_handle(output)
        else:
            print("An error edit company.")
            return error_handle("An error edit company.")
    else:
        return error_handle("company id is empty.")

#=================================


@app.route('/managercamera.html')
@login_required
@user_is("superuser")
def managercamera():
    companies = db.session.query(Companies).all()
    return render_template( 'pages/managercamera.html', companies=companies)

@app.route('/manager-camera-company.html')
@login_required
@user_is("user")
def company_managercamera():
    addresses = db.session.query(Addresses).filter(Addresses.company_id == current_user.company_id).all()
    return render_template( 'pages/manager-camera-company.html', addresses=addresses)


@app.route('/company_cameras_data')
@login_required
@user_is("user")
def company_cameras_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(Cameras.id),
        ColumnDT(Cameras.udid),
        ColumnDT(Cameras.name),
        ColumnDT(Addresses.name),
        ColumnDT(Cameras.ipaddr),
        ColumnDT(func.strftime("%Y-%m-%d %H:%M:%S", Cameras.time))
    ]

    if (current_user.has_roles("superuser")):
        query = db.session.query().select_from(Cameras).outerjoin(Addresses).filter()
    else:
        query = db.session.query().select_from(Cameras).outerjoin(Addresses).filter(Cameras.company_id == current_user.company_id)

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    
    # print(rowTable.output_result())
    return jsonify(rowTable.output_result())

@app.route('/cameras_data')
@login_required
@user_is("superuser")
def cameras_data():
    """Return server side data."""
    # defining columns

    columns = [
        ColumnDT(Cameras.id),
        ColumnDT(Cameras.udid),
        ColumnDT(Companies.name),
        ColumnDT(Cameras.ipaddr),
        ColumnDT(func.strftime("%Y-%m-%d %H:%M:%S", Cameras.time)),
        ColumnDT(Cameras.version),
        ColumnDT(Cameras.version)
    ]

    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Cameras).outerjoin(Companies).filter()

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    for i in range(len(rowTable.output_result()["data"])):
        max = db.session.query(func.max(Versions.id)).first()
        for max_ in max:
            if (rowTable.output_result()["data"][i]['6']):
                rowTable.output_result()["data"][i]['6'] = rowTable.output_result()["data"][i]['6'] < max_
            else:
                rowTable.output_result()["data"][i]['5'] = 0
                rowTable.output_result()["data"][i]['6'] = False
    print(rowTable.output_result())
    return jsonify(rowTable.output_result())

@app.route('/add_cam', methods=['POST'])
@login_required
@user_is("superuser")
def add_cam():
    output = json.dumps({"success": True})
    cam_udid =  request.form['cam_udid']
    company_id =  request.form['company']

    if company_id:
        cam = Cameras(udid= cam_udid, company_id=company_id)
        db.session.add(cam)
        db.session.commit()
        if cam:
            return success_handle(output)
        else:
            print("An error saving camera.")
            return error_handle("An error saving camera.")
    else:
        return error_handle("company_id is empty.")


@app.route('/update_firmware', methods=['POST'])
@login_required
@user_is("superuser")
def update_firmware():
    output = json.dumps({"success": True})
    cam_id =  request.form['id']
    cam = Cameras.query.filter_by(id=cam_id).first()
    if cam:
        version = Versions.query.order_by(Versions.id.desc()).first()
        if version:
            room = session.get(str(cam.company_id))
            data = {}
            data['file'] = version.file
            data['id'] = version.id
            data['camera_id'] = cam.udid
            socketio.emit('update_firmware', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )

        return success_handle(output)
    else:
        return error_handle("cam is not found.")

@app.route('/del_cam', methods=['POST'])
@login_required
@user_is("superuser")
def del_cam():
    output = json.dumps({"success": True})
    cam_id =  request.form['id']

    ret = Cameras.query.filter_by(id=cam_id).delete()
    if ret:
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete camera.")

@app.route('/edit_cam', methods=['POST'])
@login_required
@user_is("superuser")
def edit_cam():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    cam_udid =  request.form['editCameraUdid']
    cam_id =  request.form['editID']
    company_id =  request.form['editCompany']

    if cam_id:
        cam = Cameras.query.filter_by(id=cam_id)
        if cam:
            db.session.query(Cameras).filter_by(id=cam_id).update({"udid": cam_udid, "company_id": company_id}, synchronize_session='fetch')
            db.session.commit()
            room = session.get(str(company_id))
            data = {}
            data['camera_id'] = cam_udid
            socketio.emit('restart', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )
            return success_handle(output)
        else:
            print("An error edit camera.")
            return error_handle("An error edit camera.")
    else:
        return error_handle("Name is empty.")

@app.route('/edit_ccam', methods=['POST'])
@login_required
@user_is("user")
def edit_ccam():
    output = json.dumps({"success": True})
    #print(request.form['name'])    
    cam_id =  request.form['editID']
    address_id =  request.form['editAddress']
    cam_name =  request.form['editName']

    if cam_id:
        cam = Cameras.query.filter_by(id=cam_id).first()

        if cam:
            if not is_same_company(cam.company_id):
                return error_handle("An error edit camera.")
            db.session.query(Cameras).filter_by(id=cam_id).update({"address_id": address_id, "name": cam_name}, synchronize_session='fetch')
            db.session.commit()
            return success_handle(output)
        else:
            print("An error edit camera.")
            return error_handle("An error edit camera.")
    else:
        return error_handle("Name is empty.")

#-------------------------------------------------


@app.route('/del_face', methods=['POST'])
@login_required
@user_is("user")
def del_face():
    output = json.dumps({"success": True})
    face_id =  request.form['id']
    
    if face_id:
        face_array = face_id.split(",")




        db_path = feature_db_path + str(current_user.company_id) + ".db"
        if not os.path.isfile(db_path):
            db_o_path = feature_db_path + "mq_feature_empty.db"
            shutil.copy2(db_o_path, db_path)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        for id_ in face_array:
            face = Faces.query.filter_by(id=id_).first()

            user = db.session.query(User).filter(User.id == face.user_id).first() 

            if not is_same_company(user.company_id):
                return error_handle("No permission to delete face.")

            if face:
                file_name = face.file_name.replace(face_image_path, "")
                file_name = file_name.replace("/" + str(current_user.company_id), "")
                file_name = file_name.replace("/", "")
                file_name = file_name.replace(".jpg", "")
                print(file_name)

                try:
                    sql_Delete_query = """DELETE FROM Features WHERE Features.name = '{0}'""".format(file_name)
                    c.execute(sql_Delete_query)
                    Faces.query.filter_by(id=id_).delete()
                except sqlite3.IntegrityError as e:
                    print('Remove feature errror: ', e.args[0]) # column name is not unique
                except Exception as e:
                    conn.rollback()
                    conn.close()
                    db.session.rollback()
                    return error_handle("An error delete face.")

                #TODO: send socket to camera to remove
                room = session.get(str(user.company_id))
                data = {}
                data['name'] = file_name
                print(data);
                socketio.emit('feature_del', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )
                
            else:
                return error_handle("An error delete face.")

        conn.commit()
        conn.close()

        db.session.commit()
        return success_handle(output)

    return error_handle("An error delete face.")

@app.route('/profile.html')
@login_required
def profile():
    companies = []
    roles = db.session.query(Role).all()
    refined_roles = []
    
    if (current_user.has_roles("user")):
        for role in roles:
            if role.name != "superuser":
                refined_roles.append(role)
        companies = db.session.query(Companies).filter(Companies.id == current_user.company_id).all()
    elif (current_user.has_roles("superuser")):
        refined_roles = roles
        companies = db.session.query(Companies).all()

    user_id = request.args.get('id')
    if not user_id:
        user_id = current_user.id
    user = db.session.query(User).filter(User.id == user_id).outerjoin(Companies).first()
    if user:
        if (not current_user.has_roles("superuser") and user.company_id != current_user.company_id):
            return render_template( 'pages/permission_denied.html')

        if (current_user.has_roles("staff") or current_user.has_roles("user")):
            if (user.id != current_user.id):
                return render_template( 'pages/permission_denied.html')

        faces = db.session.query(Faces).filter(Faces.user_id == user_id).all()
        gender = -1
        age = -1
        if (len(faces) > 0):
            print(os.path.join(basedir, faces[0].file_name))
            img = cv2.imread(os.path.join(basedir, faces[0].file_name), cv2.IMREAD_COLOR)
            if img is not None:
                nimg1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                nimg1 = np.transpose(nimg1, (2,0,1))
                ga_model = get_model(ctx, image_size, ga_model_path, 'fc1')
                gender, age = get_ga(ga_model, nimg1)
                print(gender)
                print(age)

        
        return render_template( 'pages/profile.html', user=user, companies = companies, roles=refined_roles, faces = faces, gender = gender, age = age)
        
    else:
        return render_template( 'pages/error-404.html')


@app.route('/profile_log_time')
@login_required
def profile_log_time():
    output = json.dumps({"success": True})

    user_id = request.args.get('id')
    if not user_id:
        user_id = current_user.id
    user = db.session.query(User).filter(User.id == user_id).first()
    if user:

        columns = [
            ColumnDT(Histories.id),
            ColumnDT(Histories.image),
            ColumnDT(func.strftime("%Y-%m-%d %H:%M:%S", Histories.time)),
            ColumnDT(Cameras.name),
            ColumnDT(Addresses.name)
        ]


        query = db.session.query().select_from(Histories).join(Cameras).outerjoin(Addresses).filter(Histories.user_id == user_id).group_by(Histories.id).order_by(Histories.time.desc())

        # GET parameters
        params = request.args.to_dict()

        # instantiating a DataTable for the query and table needed
        rowTable = DataTables(params, query, columns)

        # returns what is needed by DataTable
        print(rowTable.output_result())
        return jsonify(rowTable.output_result())

        
    else:
        return error_handle("user is not existed.")

@app.route('/search_user', methods=['POST'])
@login_required
@user_is("user")
def search_user():
    users = User.query.whooshee_search(request.form['searchword'], match_substrings=True).outerjoin(Faces).filter(User.is_unknown == False).filter(User.company_id == current_user.company_id).group_by(User.id).all()
    faces = []
    for user in users: 
        face = db.session.query(Faces).filter(Faces.user_id == user.id).first()
        faces.append(face)

    #print(users)
    #print(faces)
    #searchword =  request.form['searchword']
    #search = "%{}%".format(searchword.upper())

    #users = db.session.query(User, Faces).outerjoin(Faces).filter(func.upper(User.full_name).ilike(func.upper(search)) | func.upper(User.user).ilike(func.upper(search))).filter(User.is_unknown == False).filter(User.company_id == current_user.company_id).group_by(User.id).all()
    #pprint(users)
    return render_template('pages/user_list.html', users=users, faces=faces)


@app.route('/confirm_user', methods=['POST'])
@login_required
@user_is("user")
def confirm_user():
    output = json.dumps({"success": True})
    user_id =  request.form['id']

    user = db.session.query(User).filter(User.id == user_id).first()
    if user:
        selectedIds =  request.form['selectedIds']
        selectedHIds =  request.form['selectedHIds']
        #pprint(selectedIds)
        #ids = str(selectedIds).split(',')
        ids = [int(n) for n in str(selectedIds).split(',')]
        hids = [int(n) for n in str(selectedHIds).split(',')]
        pprint(ids)
        pprint(hids)
        #db.session.query(Faces).filter(Faces.user_id.in_(ids)).update({"user_id": user.id}, synchronize_session='fetch')
        #db.session.query(Histories).filter(Histories.user_id.in_(ids)).update({"user_id": user.id}, synchronize_session='fetch')
        #db.session.query(User).filter(User.id.in_(ids)).filter(User.is_unknown == True).delete(synchronize_session='fetch')

        db.session.query(Histories).filter(Histories.id.in_(hids)).update({"user_id": user.id}, synchronize_session='fetch')  
        faces = db.session.query(Faces.id).join(User, Faces.user_id == User.id).filter(Faces.user_id.in_(ids)).filter(User.is_unknown == True).all()
        
        
        result_list = [row[0] for row in faces]
        print(result_list)
        db.session.query(Faces).filter(Faces.id.in_(result_list)).update({"user_id": user.id}, synchronize_session='fetch')

        db.session.query(User).filter(User.id.in_(ids)).filter(User.is_unknown == True).delete(synchronize_session='fetch')

        db.session.commit()
        return success_handle(output)

        # db_path = feature_db_path + str(current_user.company_id) + ".db"
        # if not os.path.isfile(db_path):
        #     db_o_path = feature_db_path + "mq_feature_empty.db"
        #     shutil.copy2(db_o_path, db_path)

        # conn = sqlite3.connect(db_path)
        # c = conn.cursor()
        
        # for face in faces:
        #     if face:
        #         file_name = face.file_name.replace(face_image_path, "")
        #         file_name = file_name.replace("/" + str(current_user.company_id), "")
        #         file_name = file_name.replace("/", "")
        #         file_name = file_name.replace(".jpg", "")
        #         print(file_name)

        #         try:
        #             sql_Delete_query = """DELETE FROM Features WHERE Features.name = '{0}'""".format(file_name)
        #             c.execute(sql_Delete_query)
        #             Faces.query.filter_by(id=face.id).delete()
        #         except sqlite3.IntegrityError as e:
        #             print('Remove feature errror: ', e.args[0]) # column name is not unique
        #         except Exception as e:
        #             conn.rollback()
        #             conn.close()
        #             db.session.rollback()
        #             return error_handle("An error delete face.")

        #         #TODO: send socket to camera to remove
        #         room = session.get(str(user.company_id))
        #         data = {}
        #         data['name'] = file_name
        #         socketio.emit('feature_del', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )
                
        #     else:
        #         return error_handle("An error delete face.")

        # conn.commit()
        # conn.close()

        # #TODO send socket

        # db.session.commit()

        # return success_handle(output)
    else:
        return error_handle("user is not existed.")

@app.route('/reset_password', methods=['POST'])
@login_required
@user_is("user")
def reset_password():
    output = json.dumps({"success": True})
    user_id =  request.form['id']

    if user_id:
        user_ = User.query.filter_by(id=user_id).first()
        
        if user_:
            if not is_same_company(user_.company_id):
                return error_handle("An error reset password.")
            newpassword = randomString()
            body = "Cảm ơn đã sử dụng dịch vụ của chúng tôi. Mật khẩu mới của bạn là " + newpassword;
            msg = Message(subject="[MQ CRM] Tạo lại mật khẩu",
              sender="crm@mqsolutions.vn",
              recipients=[user_.email], # replace with your email for testing
              body=body)
            ret = mail.send(msg)

            if not ret:
                db.session.query(User).filter_by(id=user_id).update({"password": bc.generate_password_hash(newpassword)}, synchronize_session='fetch')
                db.session.commit()
                return success_handle(output)
            else:
                print("An error reset password.")
                return error_handle("An error reset password.")
        else:
            print("An error reset password.")
            return error_handle("An error reset password.")
    else:
        return error_handle("user id is empty.")

@app.route('/edit_password', methods=['POST'])
@login_required
def edit_password():
    output = json.dumps({"success": True})
    user_id =  request.form['id']
    password =  request.form['password']
    newpassword =  request.form['newpassword']

    if user_id:
        if (int(user_id) != current_user.id):
            return error_handle("Cannot edit other user.")

        user_ = User.query.filter_by(id=user_id).first()
        if user_:
            if bc.check_password_hash(user_.password, password):
                db.session.query(User).filter_by(id=user_id).update({"password": bc.generate_password_hash(newpassword)}, synchronize_session='fetch')
                db.session.commit()
                return success_handle(output)
            else:
                print("An error edit password.")
                return error_handle("An error edit password.")
        else:
            print("An error edit password.")
            return error_handle("An error edit password.")
    else:
        return error_handle("user id is empty.")


def align_face(img, landmarks, crop_size=112):
    """Align face on the photo
    
    Arguments:
        img {PIL.Image} -- Image with face
        landmarks {np.array} -- Key points
    
    Keyword Arguments:
        crop_size {int} -- Size of face (default: {112})
    
    Returns:
        PIL.Image -- Aligned face
    """
    facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    return img_warped

@app.route('/add_history_face', methods=['POST'])
@login_required
@user_is("user")
def add_history_face():
    output = json.dumps({"success": True})
    user_id =  request.form['id']
    hid =  request.form['hid']


    user_ = User.query.filter_by(id=user_id).first()

    if (not user_):
        return error_handle("User is not existed.")

    history = Histories.query.filter_by(id=hid).first()

    if history:
        db_path = feature_db_path + str(current_user.company_id) + ".db"
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        img = cv2.imread(path.join(basedir, history.image), cv2.IMREAD_COLOR)

        nimg1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nimg1 = np.transpose(nimg1, (2,0,1))
        model = get_model(ctx, image_size, model_path, 'fc1')
        facenet_fingerprint = get_feature(model, nimg1).reshape(1,-1)

        if (len(facenet_fingerprint) > 0):
            feature_str = ""
            for value in facenet_fingerprint[0]:
                feature_str = feature_str + str(value) + "#" 
            
            face = Faces(user_id= user_id, user_id_o=user_id, file_name = history.image)
            db.session.add(face)

            try:
                file_name = face.file_name.replace(face_image_path, "")
                file_name = file_name.replace("/" + str(user_.company_id), "")
                file_name = file_name.replace("/", "")
                file_name = file_name.replace(".jpg", "")
                c.execute('''INSERT INTO Features (name, data) VALUES (?, ?)''', (file_name , feature_str,))

                room = session.get(str(current_user.company_id))
                data = {}
                data['name'] = file_name
                data['feature'] = feature_str
                socketio.emit('feature', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )

            except sqlite3.IntegrityError as e:
                print('Insert feature errror: ', e.args[0]) # column name is not unique



        conn.commit()
        conn.close()
        db.session.commit()
        
        return success_handle(output)
    else:
        return error_handle("History is not existed.")


@app.route('/edit_profile', methods=['POST'])
@login_required
@user_is("superuser")
def edit_profile():
    output = json.dumps({"success": True})
    user_id =  request.form['id']
    user =  request.form['user']
    full_name =  request.form['name']
    email =  request.form['email']
    phone =  request.form['phone']
    birth =  request.form['birth']
    role =  request.form['role']
    gender =  request.form['gender']
    #position =  request.form['position']
    company =  request.form['company']
    #code =  request.form['code']

    file_num =  request.form['file_num']
    file = None
    file_image_path = ""
    
    user_ = User.query.filter_by(id=user_id).first()

    if (not user_):
        return error_handle("User is not existed.")

    if not is_same_company(user_.company_id):
        return error_handle("Cannot edit other company's user.")

    if user_:
        # if int(file_num) > 0:
        #     for i in range(int(file_num)):
        #         file_index = 'file' + str(i)
        #         if file_index in request.files:
        #             file = request.files[file_index]
        #             if file:
        #                 if file.mimetype not in ['image/png', 'image/jpeg', 'application/octet-stream']:
        #                     print("File extension is not allowed")
        #                     return error_handle("We are only allow upload file with *.png , *.jpg")
        #                 else:
        #                     filename = str(int(time.time())) + secure_filename(file.filename)
        #                     urlSafeEncodedBytes = base64.urlsafe_b64encode(filename.encode("utf-8"))[:21]
        #                     filename_str = str(urlSafeEncodedBytes)
        #                     filename_str = filename_str.replace("=","")
        #                     filename_str = filename_str.replace("'","")

        #                     file_image_path = path.join(face_image_path + "/" + str(user_.company_id), user_id + "_" + filename_str + ".jpg");
        #                     file_image_path_no_ext = path.join(face_image_path + "/" + str(user_.company_id), user_id + "_" + filename_str);
        #                     file.save(path.join(basedir, file_image_path))

        #                     img = cv2.imread(path.join(basedir, file_image_path), cv2.IMREAD_COLOR)
                        
        #                     bboxes = detector.detect_faces(img)

        #                     i = 0
        #                     db_path = feature_db_path + str(user_.company_id) + ".db"
        #                     conn = sqlite3.connect(db_path)
        #                     c = conn.cursor()
        #                     if len(bboxes) > 0:
        #                         for bboxe in bboxes:
        #                             bbox = bboxe['box']
        #                             bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
        #                             landmarks = bboxe['keypoints']
        #                             landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
        #                                                 landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
        #                             landmarks = landmarks.reshape((2,5)).T
        #                             nimg = face_preprocess.preprocess(img, bbox, landmarks, image_size='112,112')

        #                             nimg1 = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        #                             nimg1 = np.transpose(nimg1, (2,0,1))
        #                             model = get_model(ctx, image_size, model_path, 'fc1')
        #                             facenet_fingerprint = get_feature(model, nimg1).reshape(1,-1)

        #                             if (len(facenet_fingerprint) > 0):
        #                                 feature_str = ""
        #                                 for value in facenet_fingerprint[0]:
        #                                     feature_str = feature_str + str(value) + "#"
        #                                 #print(feature_str) 

        #                                 face_img = file_image_path
        #                                 if (i > 0):
        #                                     face_img = file_image_path_no_ext + "_" + str(i) + ".jpg"
                                        
        #                                 cv2.imwrite(path.join(basedir, face_img),nimg)
        #                                 face = Faces(user_id= user_id, user_id_o=user_id, file_name = face_img)
        #                                 db.session.add(face)

        #                                 try:
        #                                     feature_name = user_id + "_" + filename_str
        #                                     if i > 0:
        #                                         feature_name = user_id + "_" + filename_str + "_" + str(i)
        #                                     c.execute('''INSERT INTO Features (name, data) VALUES (?, ?)''', (feature_name , feature_str,))
        #                                 except sqlite3.IntegrityError as e:
        #                                     print('Insert feature errror: ', e.args[0]) # column name is not unique

        #                                 i = i + 1

        #                                 room = session.get(str(current_user.company_id))
        #                                 data = {}
        #                                 data['name'] = feature_name
        #                                 data['feature'] = feature_str
        #                                 socketio.emit('feature', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )


        #                                 #TODO: send socket to camera
        #                             else:
        #                                 return error_handle("An error add images.")
        #                     else:
        #                         return error_handle("An error add images.")
        #                     conn.commit()
        #                     conn.close()
        #                     db.session.commit()
        try:
            if user and user != "":
                db.session.query(User).filter_by(id=user_id).update({"user": user}, synchronize_session='fetch')

            if email and email != "":
                db.session.query(User).filter_by(id=user_id).update({"email": email}, synchronize_session='fetch')

            if phone and phone != "":
                db.session.query(User).filter_by(id=user_id).update({"phone": phone}, synchronize_session='fetch')
            
            if gender and gender != "null":
                db.session.query(User).filter_by(id=user_id).update({"gender": gender}, synchronize_session='fetch')
            
            if full_name and full_name != "":
                db.session.query(User).filter_by(id=user_id).update({"full_name": full_name}, synchronize_session='fetch')

            # if user_.is_unknown:
            #     db.session.query(User).filter_by(id=user_id).update({"confirmed_at": datetime.now(), "is_unknown": False}, synchronize_session='fetch')

            if current_user.has_roles("superuser") and company and company != "null":
                db.session.query(User).filter_by(id=user_id).update({"company_id": company}, synchronize_session='fetch')

            # if user_.has_roles("staff") and position and position !="":
            #     db.session.query(User).filter_by(id=user_id).update({"position": position}, synchronize_session='fetch')

            # if (user_.has_roles("user") or user_.has_roles("staff"))  and code and code !="":
            #     db.session.query(User).filter_by(id=user_id).update({"code": code}, synchronize_session='fetch')
            
            if (birth):
                birth_date = datetime.strptime(birth,"%m/%d/%Y").date()
                db.session.query(User).filter_by(id=user_id).update({"birthday": birth_date}, synchronize_session='fetch')

            if role and role != "null":
                role_ = db.session.query(Role).filter_by(id=role).first()
                if role_:
                    sql = text('delete from roles_users where user_id={0}'.format(user_id))
                    result = db.session.connection().execute(sql)
                    #user_.roles = []
                    user_.roles.append(role_)

            db.session.commit()
        except SQLAlchemyError as e:
            print(e)
            db.session.rollback()
            return error_handle("An error edit profile.")
        except:
            print("Edit profile exception.")
            db.session.rollback()
            return error_handle("An error edit profile.")
        finally:
            db.session.close()
        
        return success_handle(output)
    else:
        print("An error edit profile.")
        return error_handle("An error edit profile.")

    return error_handle("An error edit profile.")

@app.route('/search-face.html')
@login_required
@user_is("user")
def searchface():
    return render_template( 'pages/search-face.html')

# Authenticate user
@app.route('/lfsearch', methods=['POST'])
def fsearch():
    return;
    image = request.files['photo']
    if image.mimetype not in ['image/png', 'image/jpeg', 'application/octet-stream']:
        print("File extension is not allowed")
        return error_handle("We are only allow upload file with *.png , *.jpg")

    image_type = image.name.split('.')[-1] #png/jpg/jpeg
    now = datetime.now()
    image_path = f'{now.day}{now.month}{now.year}_{now.hour}:{now.minute}:{now.second}.{image_type}'
    full_path = os.path.join(face_image_path, image_path)
    with open(full_path, 'wb+') as destination:
        destination.write(image.read())

        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                        
        bboxes = detector.detect_faces(img)

        if len(bboxes) > 0:
            if len(bboxes) == 1:
                bbox = bboxes[0]['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                landmarks = bboxes[0]['keypoints']
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                    landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(img, bbox, landmarks, image_size='112,112')

                nimg1 = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg1 = np.transpose(nimg1, (2,0,1))
                model = get_model(ctx, image_size, model_path, 'fc1')
                facenet_fingerprint = get_feature(model, nimg1).reshape(1,-1)
                #similarity = 1 - spatial.distance.cosine(db_feature, facenet_fingerprint)

            else:
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                        landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2,5)).T
                    nimg = face_preprocess.preprocess(img, bbox, landmarks, image_size='112,112')

                    nimg1 = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg1 = np.transpose(nimg1, (2,0,1))
                    model = get_model(ctx, image_size, model_path, 'fc1')
                    facenet_fingerprint = get_feature(model, nimg1).reshape(1,-1)
                    i = 0
                    if (len(facenet_fingerprint) > 0):

                        face_img = full_path
                        if (i > 0):
                            face_img = file_image_path_no_ext + "_" + str(i) + ".jpg"
                        
                        cv2.imwrite(path.join(basedir, face_img),nimg)
                       

                        i = i + 1

        else:
            return redirect('search-face.html')

    



@app.route('/users.html')
@login_required
@user_is("user")
def users():
    companies = []
    roles = db.session.query(Role).all()
    refined_roles = []
    
    if (current_user.has_roles("user")):
        for role in roles:
            if role.name != "admin" and role.name != "superuser":
                refined_roles.append(role)
        companies = db.session.query(Companies).filter(Companies.id == current_user.company_id).all()
    elif (current_user.has_roles("superuser")):
        refined_roles = roles
        companies = db.session.query(Companies).all()

    return render_template( 'pages/users.html', companies=companies, roles=refined_roles)

@app.route('/users_data')
@login_required
@user_is("user")
def users_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(User.id),
        ColumnDT(User.user),
        ColumnDT(Role.name),
        ColumnDT(User.full_name),
        ColumnDT(User.email),
        ColumnDT(Companies.name),
        ColumnDT(User.position),
        ColumnDT(User.code)
    ]


    if (current_user.has_roles("superuser")):
        query = db.session.query().select_from(User).join(User.roles).outerjoin(Companies).filter(User.is_unknown == False)
    else:
        query = db.session.query().select_from(User).join(User.roles).outerjoin(Companies).filter(User.is_unknown == False).filter(User.company_id == current_user.company_id).filter((Role.name == "staff") | (Role.name == "user"))

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    # print(rowTable.output_result())
    return jsonify(rowTable.output_result())

@app.route('/add_user', methods=['POST'])
@login_required
@user_is("user")
def add_user():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    
    if current_user.has_roles("superuser"):  
        company = request.form['company']     
        if not is_same_company(company):
            return error_handle("An error add user.")
    else:
        company = current_user.company_id

   
    account =  request.form['user']


    if account:
        user = User(user=account, password=bc.generate_password_hash(request.form['password']), email=request.form['email'], confirmed_at=datetime.now())
        # user.code = request.form['code']
        user.company_id = company
        # user.position = request.form['position']
        user.full_name = request.form['name']

        user_role = db.session.query(Role).filter_by(id=request.form['role']).first()
        user.roles = []
        user.roles.append(user_role)
        db.session.add(user)
        db.session.commit()

        if user.id:
            return success_handle(output)
        else:
            print("An error saving user.")
            return error_handle("An error saving user.")
    else:
        return error_handle("user is empty.")


@app.route('/del_user', methods=['POST'])
@login_required
@user_is("user")
def del_user():
    output = json.dumps({"success": True})
    user_id =  request.form['id']
    user = User.query.filter_by(id=user_id).first()
    
    if user:
        if not is_same_company(user.company_id):
            return error_handle("An error delete user.")
        user.roles = []
        db.session.commit()

        db_path = feature_db_path + str(user.company_id) + ".db"
        if not os.path.isfile(db_path):
            db_o_path = feature_db_path + "mq_feature_empty.db"
            shutil.copy2(db_o_path, db_path)

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        faces = Faces.query.filter_by(user_id=user.id).all()
        
        for face in faces:
            file_name = face.file_name.replace(face_image_path, "")
            file_name = file_name.replace("/" + str(current_user.company_id), "")
            file_name = file_name.replace("/", "")
            file_name = file_name.replace(".jpg", "")
            print(file_name)

            try:
                sql_Delete_query = """DELETE FROM Features WHERE Features.name = '{0}'""".format(file_name)
                c.execute(sql_Delete_query)
                conn.commit()
                Faces.query.filter_by(id=face.id).delete()
                Histories.query.filter_by(user_id=user.id).delete()
            except sqlite3.IntegrityError as e:
                print('Remove feature errror: ', e.args[0]) # column name is not unique
            except Exception as e:
                conn.rollback()

            #TODO: send socket to camera to remove

            room = session.get(str(user.company_id))
            data = {}
            data['name'] = file_name
            socketio.emit('feature_del', { 'data' :  json.dumps(data) }, namespace='/camera', room=room )


        conn.commit()
        conn.close()

        db.session.commit()

        ret = User.query.filter_by(id=user_id).delete()
        if ret:
            db.session.commit()
            return success_handle(output)
        else:
            return error_handle("An error delete user.")
    else:
        return error_handle("An error delete user.")


#----------------------------------------------------------
@app.route('/address_data')
@login_required
@user_is("user")
def address_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(Addresses.id),
        ColumnDT(Addresses.name),
        ColumnDT(Addresses.address),
        ColumnDT(Addresses.start),
        ColumnDT(Addresses.end),
    ]
 
    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Addresses).filter(Addresses.company_id == current_user.company_id)

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    for i in range(len(rowTable.output_result()["data"])):
        if rowTable.output_result()["data"][i]['3']:
            rowTable.output_result()["data"][i]['3'] = rowTable.output_result()["data"][i]['3'].strftime("%I:%M %p")
        if rowTable.output_result()["data"][i]['4']:
            rowTable.output_result()["data"][i]['4'] = rowTable.output_result()["data"][i]['4'].strftime("%I:%M %p")

    #print(rowTable.output_result())

    return jsonify(rowTable.output_result())

@app.route('/edit_address', methods=['POST'])
@login_required
@user_is("user")
def edit_address():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    address_id =  request.form['editID']
    name =  request.form['editName']
    address =  request.form['editAddress']
    start =  request.form['editStart']
    end =  request.form['editEnd']

    if address_id and name and address:
        address_ = Addresses.query.filter_by(id=address_id).first()

        if address_:
            if not is_same_company(address_.company_id):
                return error_handle("An error edit address.")

            db.session.query(Addresses).filter_by(id=address_id).update({"name": name, "address": address}, synchronize_session='fetch')
            db.session.commit()

            if (start):
                start_time = datetime.strptime(start,"%I:%M %p").time()
                db.session.query(Addresses).filter_by(id=address_id).update({"start": start_time}, synchronize_session='fetch')
            if (end):
                end_time = datetime.strptime(end,"%I:%M %p").time()
                db.session.query(Addresses).filter_by(id=address_id).update({"end": end_time}, synchronize_session='fetch')
            db.session.commit()

            return success_handle(output)
        else:
            print("An error edit address.")
            return error_handle("An error edit address.")
    else:
        return error_handle("Name is empty.")

@app.route('/add_address', methods=['POST'])
@login_required
@user_is("user")
def add_address():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    name =  request.form['name']
    address =  request.form['address']
    start =  request.form['start']
    end =  request.form['end']

    if name or address:
        address_ = Addresses(name=name, address=address, company_id=current_user.company_id)
        db.session.add(address_)
        db.session.commit()
        if address_:
            if (start):
                start_time = datetime.strptime(start,"%I:%M %p").time()
                db.session.query(Addresses).filter_by(id=address_.id).update({"start": start_time}, synchronize_session='fetch')
            if (end):
                end_time = datetime.strptime(end,"%I:%M %p").time()
                db.session.query(Addresses).filter_by(id=address_.id).update({"end": end_time}, synchronize_session='fetch')
            db.session.commit()
            return success_handle(output)
        else:
            print("An error saving address.")
            return error_handle("An error saving address.")
    else:
        return error_handle("Name or address is empty.")
    
@app.route('/del_address', methods=['POST'])
@login_required
@user_is("user")
def del_address():
    output = json.dumps({"success": True})
    address_id =  request.form['id']
    address = Addresses.query.filter_by(id=address_id).first()
   
    if address:  
        if not is_same_company(address.company_id):
            return error_handle("An error delete address.")
        Addresses.query.filter_by(id=address_id).delete()
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete address.")

#----------------------------------------------------------
@app.route('/role_data')
@login_required
@user_is("superuser")
def role_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(Role.id),
        ColumnDT(Role.name),
    ]

    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Role).filter()

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    # print(rowTable.output_result())
    return jsonify(rowTable.output_result())

@app.route('/edit_roles', methods=['POST'])
@login_required
@user_is("superuser")
def edit_roles():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    name =  request.form['name']
    role_id =  request.form['id']

    if name and id:
        role = Role.query.filter_by(id=role_id)
        if role:
            db.session.query(Role).filter_by(id=role_id).update({"name": name}, synchronize_session='fetch')
            db.session.commit()
            return success_handle(output)
        else:
            print("An error edit role.")
            return error_handle("An error edit role.")
    else:
        return error_handle("Name is empty.")

@app.route('/add_roles', methods=['POST'])
@login_required
@user_is("superuser")
def add_roles():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    name =  request.form['name']

    if name:
        role = Role(name=name)
        db.session.add(role)
        db.session.commit()
        if role:
            return success_handle(output)
        else:
            print("An error saving role.")
            return error_handle("An error saving role.")
    else:
        return error_handle("Name is empty.")
    
@app.route('/del_roles', methods=['POST'])
@login_required
@user_is("superuser")
def del_roles():
    output = json.dumps({"success": True})
    role_id =  request.form['id']

    ret = Role.query.filter_by(id=role_id).delete()
    if ret:
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete role.")


def day_time(type, data):
    value = []
    day = {}
    for i in range(24):
        t = ''
        if i<10:
            time_begin = data + ' 0' + str(i) + ':00:00'
            if i == 9:
                time_end   = data + ' ' + str(i+1) + ':00:00'
                t = str(i+1) + ':00'
            else:
                time_end   = data + ' 0' + str(i+1) + ':00:00'
                t = '0' + str(i+1) + ':00'
        else:
            time_begin = data + ' ' + str(i) + ':00:00'
            time_end   = data + ' ' + str(i+1) + ':00:00'
            t = str(i+1) + ':00'

        if type == 0:
            count = Histories.query.join(User).filter(User.company_id == current_user.company_id).join(User.roles).filter(Role.name == "user").filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        elif type == 1:
            count = Histories.query.join(User).filter(User.company_id == current_user.company_id).join(User.roles).filter(Role.name == "staff").filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        elif type == 2:
            count = User.query.filter((User.confirmed_at>=time_begin) &  (User.confirmed_at<=time_end)).count()
        elif type == 3:
            count = Histories.query.filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        value.append(count)
        day[t] = count
    return day

def stringtodate(str):
    return datetime.strptime(str,'%Y-%m-%d')

def PlusDay(date):
    date = stringtodate(date)
    date = date + timedelta(days=1)
    return date.strftime('%Y-%m-%d')

def days_time(type, start, end):
    value = []
    days = {}
    data = start
    fis = PlusDay(end)
    while data != fis:
        time_begin = data + ' 00:00:00'
        time_end   = data + ' 23:59:59'
        if type == 0:
            count = Histories.query.join(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(Role.name == "user").filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        elif type == 1:
            count = Histories.query.join(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(Role.name == "staff").filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        elif type == 2:
            count = User.query.filter(User.confirmed_at>=time_begin) &  ((User.confirmed_at<=time_end)).count()
        elif type == 3:
            count = Histories.query.filter((Histories.time>=time_begin) &  (Histories.time<=time_end)).count()
        days[data]=count
        data = PlusDay(data)
    
    return days



@app.route('/chart', methods=['GET','POST'])
@login_required
@user_is("user")
def chart():
    start =  request.form['start']
    end =  request.form['end']
    label =  request.form['label']
    if current_user.has_roles("superuser"):
        if start == end:
            return json.dumps({
                "user": day_time(2, end),
                "history":  day_time(3, end)
            })
        else:
            return json.dumps({
                "user": days_time(2, start,end),
                "history":  days_time(3, start,end)
            })
    elif current_user.has_roles("user"):
        if start == end:
            return json.dumps({
                "visitor": day_time(0, end),
                "staff":  day_time(1, end)
            })
        else:
            return json.dumps({
                "visitor": days_time(0, start,end),
                "staff":  days_time(1, start,end)
            })

# App main route + generic routing
@app.route('/', defaults={'path': 'index.html'})
@login_required
@user_is("user")
def index(path):
    if current_user.has_roles("superuser"):
        camera_num = db.session.query(Cameras).count()
        history_num = db.session.query(Histories).join(Cameras, Cameras.id == Histories.camera).count()
        user_num = db.session.query(User).filter(User.is_unknown == False).count()
        company_num = db.session.query(Companies).count()
        return render_template( 'pages/index.html', camera_num=camera_num, history_num=history_num, user_num=user_num, company_num=company_num )
    elif current_user.has_roles("user"):
        camera_num = db.session.query(Cameras).filter(Cameras.company_id == current_user.company_id).count()
        history_num = db.session.query(Histories).join(Cameras, Cameras.id == Histories.camera).join(User, Histories.user_id == User.id).filter(User.company_id == current_user.company_id).count()
        user_num = db.session.query(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(Role.name == "user").filter(User.is_unknown == False).count()
        staff_num = db.session.query(User).join(User.roles).filter(User.company_id == current_user.company_id).filter(Role.name == "staff").filter(User.is_unknown == False).count()
        return render_template( 'pages/index.html', camera_num=camera_num, history_num=history_num, user_num=user_num, staff_num=staff_num )
    else:
        return render_template( 'pages/permission_denied.html')

@app.route('/<path>')
@login_required
@user_is("user")
def custom(path):
    try:
        # try to match the pages defined in -> pages/<input file>
        return render_template( 'pages/'+path )
    except:
        return render_template( 'pages/error-404.html' )

# Return sitemap 
@app.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')

language = 'vi_VN'

@babel.localeselector
def get_locale():
	if language != '':
		return language
	# will return language code (en/es/etc).
	return request.accept_languages.best_match(Config.LANGUAGES.keys())
	
@app.route('/translate', methods=['POST'])
def translate():
	lang = request.get_json()
	globals()['language'] = lang['lang']
	get_locale()
	return ''

@app.route('/payment', methods=['GET','POST'])
def payment():
    if request.method == 'POST':
        # Process input data and build url payment
        order_type = request.form['order_type']
        order_id = request.form['order_id']+ ":" + order_type + ":" + str(datetime.now())
        amount = request.form['amount']
        order_desc = request.form['order_desc']
        bank_code = request.form['bank_code']
        language = request.form['language']
        ipaddr = get_client_ip(request)

        vnp = vnpay()
        vnp.requestData['vnp_Version'] = '2.0.0'
        vnp.requestData['vnp_Command'] = 'pay'
        vnp.requestData['vnp_TmnCode'] = Config.VNPAY_TMN_CODE
        vnp.requestData['vnp_Amount'] = str(100 * int(amount))
        vnp.requestData['vnp_CurrCode'] = 'VND'
        vnp.requestData['vnp_TxnRef'] = order_id
        vnp.requestData['vnp_OrderInfo'] = order_desc
        vnp.requestData['vnp_OrderType'] = order_type
            # Check language, default: vn
        if language and language != '':
            vnp.requestData['vnp_Locale'] = language
        else:
            vnp.requestData['vnp_Locale'] = 'vn'
                # Check bank_code, if bank_code is empty, customer will be selected bank on VNPAY
        if bank_code and bank_code != "":
            vnp.requestData['vnp_BankCode'] = bank_code

        vnp.requestData['vnp_CreateDate'] = datetime.now().strftime('%Y%m%d%H%M%S')  # 20150410063022
        vnp.requestData['vnp_IpAddr'] = ipaddr
        vnp.requestData['vnp_ReturnUrl'] = Config.VNPAY_RETURN_URL
        vnpay_payment_url = vnp.get_payment_url(Config.VNPAY_PAYMENT_URL, Config.VNPAY_HASH_SECRET_KEY)
        if request.is_xhr:
            # Show VNPAY Popup
            result = jsonify({'code': '00', 'Message': 'Init Success', 'data': vnpay_payment_url})
            return result
        else:
            # Redirect to VNPAY
            return redirect(vnpay_payment_url)

@app.route('/payment_return', methods=['GET','POST'])
def payment_return():
    inputData = request.args
    if inputData:
        vnp = vnpay()
        vnp.responseData = inputData.to_dict()
        order_id = inputData['vnp_TxnRef']
        amount = int(inputData['vnp_Amount']) / 100
        order_desc = inputData['vnp_OrderInfo']
        vnp_TransactionNo = inputData['vnp_TransactionNo']
        vnp_ResponseCode = inputData['vnp_ResponseCode']
        vnp_TmnCode = inputData['vnp_TmnCode']
        vnp_PayDate = inputData['vnp_PayDate']
        vnp_BankCode = inputData['vnp_BankCode']
        vnp_CardType = inputData['vnp_CardType']
        if vnp.validate_response(Config.VNPAY_HASH_SECRET_KEY):
            if vnp_ResponseCode == '00':
                com = Companies.query.filter_by(id=current_user.company_id).first()
                if 'Basic package' in order_desc or 'Gói tiêu chuẩn' in order_desc:
                    com.plan_id = 2
                    db.session.commit()
                if 'Advanced package' in order_desc or 'Gói nâng cao' in order_desc:
                    com.plan_id = 3
                    db.session.commit()
                return render_template( 'pages/payment_return.html', title="Kết quả thanh toán",
                                                               result= "Thành công", order_id= order_id,
                                                               amount= amount,
                                                               order_desc= order_desc,
                                                               vnp_TransactionNo= vnp_TransactionNo,
                                                               vnp_ResponseCode= vnp_ResponseCode)
            else: 
                return render_template( 'pages/payment_return.html', title= "Kết quả thanh toán",
                                                               result= "Lỗi", order_id= order_id,
                                                               amount= amount,
                                                               order_desc= order_desc,
                                                               vnp_TransactionNo= vnp_TransactionNo,
                                                               vnp_ResponseCode= vnp_ResponseCode)
        else:
            return render_template('pages/payment_return.html',
                          title= "Kết quả thanh toán", result= "Lỗi", order_id= order_id, amount= amount,
                           order_desc= order_desc, vnp_TransactionNo= vnp_TransactionNo,
                           vnp_ResponseCode= vnp_ResponseCode, msg= "Sai checksum")
    else:
        return render_template(request, "payment_return.html", {"title": "Kết quả thanh toán", "result": ""})

@app.route('/get_client_ip', methods=['GET','POST'])
def get_client_ip(self):
    if 'HTTP_X_FORWARDED_FOR' in self.environ:
        ip = self.environ['HTTP_X_FORWARDED_FOR'].split(',')
    elif 'REMOTE_ADDR' in self.environ:
        ip = self.environ['REMOTE_ADDR']
    return ip

@app.route('/update-package.html', methods=['GET','POST'])
def update_package():
    com = Companies.query.filter_by(id=current_user.company_id).first()
    return render_template( 'pages/update-package.html', com=com)


upload_max_size = 2000

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'dataset'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='image/*, .zip, .txt, .csv, .xlsx',
    DROPZONE_MAX_FILE_SIZE=upload_max_size,
    #DROPZONE_MAX_FILES=1,
    #DROPZONE_IN_FORM=True,
    #DROPZONE_UPLOAD_ON_CLICK=True,
    #DROPZONE_UPLOAD_ACTION='handle_form',  # URL or endpoint
    #DROPZONE_UPLOAD_BTN_ID='submit',
    DROPZONE_FILE_TOO_BIG="Kích thước tệp quá lớn {{filesize}} MB. Kích thước tối đa: {{maxFilesize}} MB.",
    DROPZONE_DEFAULT_MESSAGE="Thả dữ liệu vào đây. Kích thước tối đa " + str(upload_max_size) + " MB"
)



@app.route('/uploads', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(DATASET_DIR, f.filename))

    return 'upload template'

ALLOWED_EXTENSIONS_ML = set(['csv', 'xlsx'])
ALLOWED_EXTENSIONS_DL = set(['zip'])
ALLOWED_EXTENSIONS_ST = set(['txt'])

def get_file_ext(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower()

def allowed_dataset_type(filename, type):
    if type == 0:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_ML
    elif type == 1:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_DL
    elif type == 2:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_ST
    else:
        return False

@app.route('/add-data.html')
@login_required
def add_data_page():
    
    user_id = current_user.id
    user = db.session.query(User).filter(User.id == user_id).outerjoin(Companies).first()
    if user:    
        return render_template( 'pages/add-data.html', user=user)
        
    else:
        return render_template( 'pages/error-404.html')


@app.route('/del_data', methods=['POST'])
@login_required
@user_is("user")
def del_data():
    output = json.dumps({"success": True})
    data_id =  request.form['id']
    dataset = Dataset.query.filter_by(id=data_id).first()
    if dataset:  
        Dataset.query.filter_by(id=data_id).delete()
        db.session.commit()
        return success_handle(output)
    else:
        return error_handle("An error delete address.")


@app.route('/dataset_data')
@login_required
@user_is("user")
def dataset_data():
    """Return server side data."""
    # defining columns
    columns = [
        ColumnDT(Dataset.id),
        ColumnDT(Dataset.name),
        ColumnDT(Dataset.description),
        ColumnDT(Dataset.path),
        ColumnDT(Dataset.datatype),
        ColumnDT(Dataset.time),
    ]
 
    # defining the initial query depending on your purpose
    query = db.session.query().select_from(Dataset).filter(Dataset.user_id == current_user.id)

    # GET parameters
    params = request.args.to_dict()

    # instantiating a DataTable for the query and table needed
    rowTable = DataTables(params, query, columns)

    # returns what is needed by DataTable
    for i in range(len(rowTable.output_result()["data"])):
        if rowTable.output_result()["data"][i]['5']:
            rowTable.output_result()["data"][i]['5'] = rowTable.output_result()["data"][i]['5'].strftime("%Y-%m-%d　%I:%M %p")

    #print(rowTable.output_result())

    return jsonify(rowTable.output_result())


@app.route('/add_data', methods=['POST'])
@login_required
@user_is("user")
def add_data():
    output = json.dumps({"success": True})
    #print(request.form['name'])
    dataname =  request.form['name']
    description =  request.form['description']
    datatype =  request.form['datatype']
    user_id =  request.form['user_id']
    print(dataname)
    print(description)
    print(datatype)
    print(user_id)

    if dataname:
        if 'file' in request.files:
            file = request.files['file']

            if file:
                filename = secure_filename(file.filename)
                mime_type = file.content_type
                if not allowed_dataset_type(file.filename, int(datatype)):
                    print("File extension is not allowed")
                    return error_handle("File extension is not allowed")
                else:
                    try:
                        user_path = get_user_path()

                        file_path = os.path.join(user_path, file.filename)
                        file.save(file_path)
                        if get_file_ext(file.filename) == "zip" and zipfile.is_zipfile(file_path):
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                # Get list of files names in zip
                                filenames = zip_ref.namelist()
                                # Iterate over the list of file names in given list & print them
                                for filename in filenames:
                                    print(filename)
                                    zip_ref.extract(filename, os.path.join(user_path, dataname))
                                #zip_ref.extractall(DATASET_DIR)
                            src = os.path.join(user_path, dataname)
                            dest = os.path.join(user_path, dataname + "_hub")
                            #os.rename(file_path, src)
                            ds = hub.ingest(src, dest, overwrite=True)
                            # remove zip file
                            os.remove(file_path)
                            file_path = dest
                            print(ds.summary())

                            # img = ds.images[0].numpy()  
                            # print(img)

                        dataset_ = Dataset(user_id=user_id, name=dataname, description=description, datatype=datatype, path=file_path, time=datetime.now())  
                        if dataset_:
                            db.session.add(dataset_)
                            db.session.commit()
                            return success_handle(output)
                        else:
                            print("An error saving dataset.")
                            return error_handle("An error saving dataset.")
                    except SQLAlchemyError as e:
                        print(e)
                        db.session.rollback()
                        return error_handle("An error saving dataset to database.")
                    except:
                        print("Unknown error saving dataset.")
                        db.session.rollback()
                        return error_handle("Unknown error saving dataset.")
                    finally:
                        db.session.close()
                    

                    return success_handle(output)
        else:
            return error_handle("File not found")
    else:
        return error_handle("Name is empty.")
    #if f.filename.split('.')[1] != 'png':

@app.route('/manager-training.html')
@login_required
def manager_training():
    user_id = current_user.id
    dataset = db.session.query(Dataset).filter(Dataset.user_id == user_id).filter(Dataset.datatype == 0).all()

    if dataset:
        
        return render_template( 'pages/manager-training.html', dataset=dataset)
        
    else:
        return render_template( 'pages/error-404.html')

#hoc may
@app.route('/eda', methods=['POST'])
@login_required
@user_is("user")
def eda():
    print("####################  eda() ###############################################################################################################################")
    #print(request.form['name'])
    #modeltype =  request.form['modeltype']
    data_id =  request.form['data_id']

    #print(modeltype)
    print(data_id)
    dataset = db.session.query(Dataset).filter(Dataset.id == data_id).filter(Dataset.datatype == 0).first()
    if dataset:
        df_train = pd.read_csv(dataset.path)
        profile = ProfileReport(
            df_train, title="Profiling Report for the train dataset"
        )
        user_path = get_user_path()
        eda_file_path = os.path.join(user_path, "eda.html")
        profile.config.html.navbar_show = False
        profile.to_file(eda_file_path)
        return_output = json.dumps({"success": True, "eda_file_path": eda_file_path})  
        return success_handle(return_output)
    else:
        return error_handle("Please select modeltype and dataname.")
    #if f.filename.split('.')[1] != 'png':

@app.route('/select_data', methods=['POST'])
@login_required
@user_is("user")
def select_data():
    print("####################  select_data() ###############################################################################################################################")
    #print(request.form['name'])
    #modeltype =  request.form['modeltype']
    data_id =  request.form['data_id']

    #print(modeltype)
    print(data_id)
    dataset = db.session.query(Dataset).filter(Dataset.id == data_id).filter(Dataset.datatype == 0).first()
    if dataset:
        df_train = pd.read_csv(dataset.path)
        columns_name = df_train.columns.tolist()  
        categorical_columns = df_train.select_dtypes(include=['category','object']).columns.tolist()
        return_output = json.dumps({"success": True,"columns_name": columns_name, "categorical_columns": categorical_columns})  
        return success_handle(return_output)
    else:
        return error_handle("Please select modeltype and dataname.")
    #if f.filename.split('.')[1] != 'png':


@app.route('/preprocessing_data', methods=['POST'])
@login_required
@user_is("user")
def preprocessing_data():
    print("####################  preprocessing_data() ###############################################################################################################################")
    project = request.form['project']
    modelname = request.form['modelname']
    modeltype = int(request.form['modeltype'])
    data_id = request.form['data_id']
    target_column = request.form['targetcol']
    ignore_columns = request.form['removecol'].split(',') if request.form['removecol'] != "" else None
    trainsize = float(request.form['trainsize'])
    data_split_stratify = True if request.form['data_split_stratify'] == "1" else False
    fold_strategy = request.form['crossvalidate']
    foldnum = int(request.form['foldnum'])
    numeric_imputation = request.form['missingval']
    normalize = True if request.form['nomalization'] == "1" else False
    normalize_method = request.form['nomalization_method']
    transformation = True if request.form['transformation'] == "1" else False
    transformation_method = request.form['transformation_method']
    transform_target = True if request.form['target_transformation'] == "1" else False
    transform_target_method = request.form['target_transformation_method']
    fix_imbalance = True if request.form['fix_imbalance'] == "1" else False
    categorical_imputation = request.form['cmissingval']
    unknown_categorical_method = request.form['unknown_cat_value']
    combine_rare_levels = True if request.form['ccomrarelevel'] == "1" else False
    rare_level_threshold = float(request.form['ccomrarelevelthre']) 
    feature_interaction = True if request.form['interaction'] == "1" else False
    feature_ratio = True if request.form['calratios'] == "1" else False
    polynomial_features = True if request.form['polcom'] == "1" else False
    polynomial_degree = int(request.form['poldegree'])
    polynomial_threshold = float(request.form['polthre'])
    trigonometry_features = True if request.form['trigono'] == "1" else False
    group_features = request.form['relatefeatures'].split(',') if request.form['relatefeatures'] != "" else None
    bincom = True if request.form['bincom'] == "1" else False
    select_bin_numeric_features = request.form['bincom_option'].split(',') if request.form['bincom_option'] != "" else None
    feature_selection = True if request.form['subset'] == "1" else False
    feature_selection_threshold = float(request.form['subsetthre'])
    remove_multicollinearity = True if request.form['featuredrop'] == "1" else False
    multicollinearity_threshold = float(request.form['featuredropthre'])
    remove_perfect_collinearity = True if request.form['reppecoll'] == "1" else False
    pca = True if request.form['pca'] == "1" else False
    pca_method = request.form['pca_method']
    pca_components = float(request.form['pca_keep'])
    ignore_low_variance = True if request.form['removecat'] == "1" else False
    create_clusters = True if request.form['add_cluster'] == "1" else False
    cluster_iter = int(request.form['add_cluster_thre'])
    remove_outliers = True if request.form['remove_outliers_pca'] == "1" else False
    outliers_threshold = float(request.form['remove_outliers_pca_per'])

    # print(modeltype) #
    # print(data_id) #
    # print(target_column) #
    # print(ignore_columns) 
    # print(trainsize)
    # print(fold_strategy)
    # print(foldnum)
    # print(numeric_imputation)
    # print(normalize)
    # print(normalize_method)
    # print(transformation)
    # print(transformation_method)
    # print(transform_target)
    # print(transform_target_method)
    # print(categorical_imputation)
    # print(unknown_categorical_method)
    # print(combine_rare_levels)
    # print(rare_level_threshold)
    # print(feature_interaction)
    # print(feature_ratio)
    # print(polynomial_features)
    # print(polynomial_degree)
    # print(polynomial_threshold)
    # print(trigonometry_features)
    # print(group_features)
    # print(bincom)
    # print(select_bin_numeric_features)
    # print(feature_selection)
    # print(feature_selection_threshold)
    # print(remove_multicollinearity)
    # print(multicollinearity_threshold)
    # print(remove_perfect_collinearity)
    # print(pca)
    # print(pca_method)
    # print(pca_components)
    # print(ignore_low_variance)
    # print(create_clusters)
    # print(cluster_iter)
    # print(remove_outliers)
    # print(outliers_threshold)

    is_setup = False
    dataset = db.session.query(Dataset).filter(Dataset.id == data_id).filter(Dataset.datatype == 0).first()
    if dataset:
        df = pd.read_csv(dataset.path)
        result = None
        if modeltype == 0 or modeltype == 1:
            #experiment_name = project + "_" + modelname + "_" + data_id + datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = project + "_" + modelname + "_" + data_id
            if modeltype == 0:
                from pycaret.classification import  setup, pull
                setup(data=df, target=target_column,train_size=trainsize, preprocess=True,
                    categorical_imputation=categorical_imputation, numeric_imputation=numeric_imputation,
                    normalize=normalize,normalize_method=normalize_method, transformation=transformation,
                    transformation_method=transformation_method, 
                    unknown_categorical_method=unknown_categorical_method,
                    combine_rare_levels=combine_rare_levels,rare_level_threshold=rare_level_threshold,
                    feature_interaction=feature_interaction,ignore_features=ignore_columns,
                    feature_ratio=feature_ratio,polynomial_features=polynomial_features,
                    polynomial_degree=polynomial_degree,polynomial_threshold=polynomial_threshold,
                    trigonometry_features=trigonometry_features,group_features=group_features,
                    bin_numeric_features=select_bin_numeric_features,feature_selection=feature_selection,
                    feature_selection_threshold=feature_selection_threshold,remove_multicollinearity=remove_multicollinearity,
                    multicollinearity_threshold = multicollinearity_threshold,           
                    remove_perfect_collinearity=remove_perfect_collinearity,
                    fix_imbalance = fix_imbalance, 
                    data_split_stratify=data_split_stratify,fold_strategy=fold_strategy,fold=int(foldnum),
                    pca=pca,pca_method=pca_method,pca_components=pca_components,
                    ignore_low_variance=ignore_low_variance,create_clusters=create_clusters,cluster_iter=cluster_iter,
                    remove_outliers=remove_outliers,outliers_threshold=outliers_threshold,html=False,silent=True, log_experiment=True, experiment_name=experiment_name)

            elif modeltype == 1:
                from pycaret.regression import  setup, pull
                setup(data=df, target=target_column,train_size=trainsize, preprocess=True,
                        fold_strategy=fold_strategy,fold=int(foldnum),
                        categorical_imputation=categorical_imputation, numeric_imputation=numeric_imputation,
                        normalize=normalize,normalize_method=normalize_method, transformation=transformation,
                        transformation_method=transformation_method, transform_target=transform_target,
                        transform_target_method=transform_target_method,unknown_categorical_method=unknown_categorical_method,
                        combine_rare_levels=combine_rare_levels,rare_level_threshold=rare_level_threshold,
                        feature_interaction=feature_interaction,ignore_features=ignore_columns,
                        feature_ratio=feature_ratio,polynomial_features=polynomial_features,
                        polynomial_degree=polynomial_degree,polynomial_threshold=polynomial_threshold,
                        trigonometry_features=trigonometry_features,group_features=group_features,
                        bin_numeric_features=select_bin_numeric_features,feature_selection=feature_selection,
                        feature_selection_threshold=feature_selection_threshold,remove_multicollinearity=remove_multicollinearity,
                        multicollinearity_threshold = multicollinearity_threshold,           
                        remove_perfect_collinearity=remove_perfect_collinearity,
                        pca=pca,pca_method=pca_method,pca_components=pca_components,
                        ignore_low_variance=ignore_low_variance,create_clusters=create_clusters,cluster_iter=cluster_iter,
                        remove_outliers=remove_outliers,outliers_threshold=outliers_threshold,html=False,silent=True, log_experiment=True, experiment_name=experiment_name)
            elif modeltype == 2:
                from pycaret.clustering import  setup, pull

            # setup(data = df, target = target_column, session_id=123)
            
            result = pull(True)
            if result:
                result = result.data
                print(result)

            is_setup = True
            

            # best = compare_models(exclude=['xgboost'],fold=None, cross_validation=True)
            # compare_models = pull(True)
        return_output = json.dumps({"success": True, "result": result.to_json()}) 

        if is_setup:
            from pycaret.regression import models as regression_models
            from pycaret.classification import models as classification_models
            from pycaret.clustering import models as clustering_models

            
            regression_models = regression_models() 
            classification_models = classification_models()
            clustering_models = clustering_models()
            print(regression_models)
            return_output = json.dumps({"success": True, "result": result.to_json(), "regression_models": regression_models.to_json(), "classification_models": classification_models.to_json(), "clustering_models": clustering_models.to_json()}) 

        return success_handle(return_output)
    else:
        return error_handle("Please select modeltype and dataname.")
    #if f.filename.split('.')[1] != 'png':

@app.route('/trainmodel', methods=['POST'])
@login_required
@user_is("user")
def trainmodel():
    print("####################  trainmodel() ###############################################################################################################################")
    training_enable_cross_val = True if request.form['training_enable_cross_val'] == "1" else False
    bestname = request.form['bestname']
    modeltype = int(request.form['modeltype'])

    if modeltype == 0:
        from pycaret.classification import  create_model, pull, plot_model, finalize_model, save_model
    elif modeltype == 1:
        from pycaret.regression import  create_model, pull, plot_model, finalize_model, save_model
    elif modeltype == 2:
        from pycaret.clustering import  create_model, pull, plot_model, finalize_model, save_model


    trained_model = create_model(estimator=bestname, fold=None, cross_validation=training_enable_cross_val)
    create_model = pull(True)
    #print(create_model)
    #plot = plot_model(trained_model, plot = 'residuals', save = True)
    #print(plot)
    finalized_model = finalize_model(trained_model)
    _,name = save_model(finalized_model, "out")
    print(name)
    return_output = json.dumps({"success": True, "result": create_model.to_json()}) 
    return success_handle(return_output)


@app.route('/comparemodel', methods=['POST'])
@login_required
@user_is("user")
def comparemodel():
    print("####################  comparemodel() ###############################################################################################################################")
    compare_enable_cross_val = True if request.form['compare_enable_cross_val'] == "1" else False
    sort = request.form['sort']
    modeltype = int(request.form['modeltype'])

    if modeltype == 0:
        from pycaret.classification import  compare_models, pull
    elif modeltype == 1:
        from pycaret.regression import  compare_models, pull
    elif modeltype == 2:
        from pycaret.clustering import  compare_models, pull

    best = compare_models(exclude=['xgboost'],fold=None, cross_validation=compare_enable_cross_val, sort=sort)
    compare_models = pull(True)

    return_output = json.dumps({"success": True, "result": compare_models.to_json(), "bestname":best.__class__.__name__}) 
    print("best name:" + best.__class__.__name__)
    return success_handle(return_output)


@app.route('/detail-dataset.html')
@login_required
@user_is("user")
def detaildataset():
    print("####################  detaildataset() ###############################################################################################################################")
    data_id = request.args.get('id')
    dataset = db.session.query(Dataset).filter(Dataset.id == data_id).filter(Dataset.datatype == 0).first()
    if dataset:
        #try:
            with open(dataset.path, 'rb') as f:
                enc = chardet.detect(f.read())  # or readline if the file is large
                print(enc)
            df = pd.read_csv(dataset.path,encoding = enc['encoding'])
            columns_name = df.columns.tolist() 
            table_data = df.to_html(table_id="dataTb", classes=["table", "table-striped", "table-hover", "table-head-bg-primary","nowrap"])
            return render_template('pages/detail-dataset.html', table_data=table_data, columns_name=columns_name)
        # except:
        #     return render_template('pages/detail-dataset.html')
        # finally:
        #     return render_template('pages/detail-dataset.html')
    else:
        return render_template( 'pages/error-404.html')




@app.route('/time_series_ngoc.html')
@login_required
@user_is("user")
def ngoc():
    user_id = current_user.id
    dataset = db.session.query(Dataset).filter(Dataset.user_id == user_id).filter(Dataset.datatype == 0).all()

    if dataset:
        
        return render_template( 'pages/time_series_ngoc.html', dataset=dataset)
        
    else:
        return render_template( 'time_series_ngoc.html')

@app.route('/resetpassword.html')
def resetpassword():
    return render_template('pages/resetpassword.html')

@app.route('/sendresetpassword', methods=['POST'])
def sendresetpassword():
    return render_template('pages/resetpassword.html')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=3456, debug=True)
    socketio.run(app,host='0.0.0.0', port=1309, debug=True)
