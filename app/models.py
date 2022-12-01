# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from app         import app, db, bc, whooshee
from flask_login import UserMixin
import os
import string
import random
from  configuration import Config
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import date, timedelta, datetime
from sqlalchemy_serializer import SerializerMixin
from sqlalchemy.orm import validates

roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class Role(db.Model, SerializerMixin):
    serialize_only = ('name',) 

    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __str__(self):
        return self.name

@whooshee.register_model('full_name')
class User(UserMixin, db.Model, SerializerMixin):


    serialize_only = ('id','user','email','position','full_name','code','phone', 'birthday','gender','is_unknown','faces', 'roles',) 


    id       = db.Column(db.Integer,     primary_key=True)
    user     = db.Column(db.String(64),  unique = True)
    email    = db.Column(db.String(100), nullable = True)
    password = db.Column(db.String(100), nullable=True)
    position = db.Column(db.String(100))
    full_name = db.Column(db.String(100), nullable=True)
    code = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    birthday = db.Column(db.DateTime())
    gender = db.Column(db.Integer, default=0)
    active = db.Column(db.Boolean(), default=False)
    is_unknown = db.Column(db.Boolean(), default=False)
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))
    faces = db.relationship('Faces', backref=db.backref('users', lazy=True))
    histories = db.relationship('Histories', backref=db.backref('users', lazy=True))
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=True)

    # def __init__(self, user, email, password):
    #     self.user       = user
    #     self.password   = password
    #     self.email      = email

    def __repr__(self):
        return str(self.id) + ' - ' + str(self.user)

    def has_roles(self, *args):
        return set(args).issubset({role.name for role in self.roles})
        
    def save(self):

        # inject self into db session    
        db.session.add ( self )

        # commit change and save the object
        db.session.commit( )

        return self 

    @validates('code', 'name')
    def convert_upper(self, key, value):
        return value.upper()


class Plans(db.Model, SerializerMixin):
    __tablename__ = 'plans'
    serialize_only = ('id','name','company',)  

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    company = db.relationship('Companies', backref=db.backref('plans', lazy=True))

@whooshee.register_model('name')
class Companies(db.Model, SerializerMixin):
    __tablename__ = 'companies'
    serialize_only = ('id','name','email','phone','address','logo_image','secret',)   
    #serialize_rules = ('-related_models.companies',)

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False,  unique = True)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    logo_image = db.Column(db.String(255), nullable=True)
    secret = db.Column(db.String(255), nullable=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('plans.id'), nullable=True)
    users = db.relationship('User', backref=db.backref('companies', lazy=True))
    camera = db.relationship('Cameras', backref=db.backref('companies', lazy=True))
    addresses = db.relationship('Addresses', backref=db.backref('companies', lazy=True))

    def __str__(self):
        return self.name

class Cameras(db.Model, SerializerMixin):
    __tablename__ = 'cameras'

    serialize_only = ('id','name','udid','ipaddr','time','address_id', 'online')   

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    udid = db.Column(db.String(32), nullable=False)
    name = db.Column(db.String(255), nullable=True)
    ipaddr = db.Column(db.String(16), nullable=True)
    time = db.Column(db.DateTime(), nullable=True)
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=True)
    address_id = db.Column(db.Integer, db.ForeignKey('addresses.id'), nullable=True)
    version = db.Column(db.Integer, db.ForeignKey('versions.id'), nullable=True)
    histories = db.relationship('Histories', backref=db.backref('cameras', lazy=True))
    
    def __str__(self):
        return self.udid

    @hybrid_property
    def online(self):
        if  self.time:
            time_left = datetime.now() - self.time
            if (time_left.total_seconds() < 360):
                return True
            else:
                return False
        else:
            return False


class Addresses(db.Model, SerializerMixin):
    __tablename__ = 'addresses'

    serialize_only = ('id','name','address','start','end', 'camera',) 

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    start = db.Column(db.Time(), nullable=True)
    end = db.Column(db.Time(), nullable=True)

    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=True)
    camera = db.relationship('Cameras', backref=db.backref('addresses', lazy=True))

    def __str__(self):
        return self.name

class Faces(db.Model, SerializerMixin):
    __tablename__ = 'faces'

    serialize_only = ('id','file_name',) 

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_id_o = db.Column(db.Integer, nullable=True)
    file_name = db.Column(db.String(255), nullable=False)


class Histories(db.Model, SerializerMixin):
    __tablename__ = 'histories'

    serialize_only = ('id','user_id','image','time','camera','face',) 

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_id_o = db.Column(db.Integer, nullable=True)
    image = db.Column(db.String(255), nullable=False)
    time = db.Column(db.DateTime(), nullable=False)
    camera = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=True)

    @hybrid_property
    def face(self):
        if self.user_id:
            face = db.session.query(Faces).filter(Faces.user_id == self.user_id).first()
            if (face):
                return face.file_name
            else:
                return None
        else:
            return None

class Versions(db.Model):
    __tablename__ = 'versions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    file = db.Column(db.String(255), nullable=False)
    confirmed_at = db.Column(db.DateTime(timezone=True), default=datetime.now)


class Project(db.Model, SerializerMixin):
    __tablename__ = 'projects'

    serialize_only = ('id','name','user_id','description') 

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(1000), nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def __str__(self):
        return self.name

class Dataset(db.Model, SerializerMixin):
    __tablename__ = 'dataset'

    serialize_only = ('id','user_id', 'name', 'description', 'path', 'datatype', 'time') 

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    name = db.Column(db.String(255), nullable=False,  unique = True)
    description = db.Column(db.String(255), nullable=False,  unique = False)
    path = db.Column(db.String(255), nullable=False,  unique = False)
    datatype = db.Column(db.Integer, nullable=True)
    time = db.Column(db.DateTime(), nullable=False)
