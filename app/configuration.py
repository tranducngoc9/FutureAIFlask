# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

import os

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config():

	CSRF_ENABLED = True
	SECRET_KEY   = "77tgFCdrEEdv77554##@3" 
	
	SQLALCHEMY_TRACK_MODIFICATIONS 	= False

	SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')
	#SQLALCHEMY_DATABASE_URI = 'mysql://username:password@server/db')

	LANGUAGES = {
		'vi_VN': 'Việt Nam',
		'en': 'English',
		'es': 'Español',
		'fr': 'Française',
		'de':'Deutsche',
		'ja_JP': '日本語'
	}

	LANGUAGE_CODE = 'en-us'

	TIME_ZONE = 'Asia/Saigon'

	USE_I18N = True

	USE_L10N = True

	USE_TZ = True

	WHOOSHEE_MIN_STRING_LEN = 1
	# Static files (CSS, JavaScript, Images)
	# https://docs.djangoproject.com/en/1.10/howto/static-files/
	STATICFILES_DIRS = (
		os.path.join(BASE_DIR, 'static/'),
	)
	STATIC_URL = '/static/'
	# VNPAY CONFIG
	VNPAY_RETURN_URL = 'http://localhost:1309/payment_return'  # get from config
	VNPAY_PAYMENT_URL = 'http://sandbox.vnpayment.vn/paymentv2/vpcpay.html'  # get from config
	VNPAY_API_URL = 'http://sandbox.vnpayment.vn/merchant_webapi/merchant.html'
	VNPAY_TMN_CODE = 'IAM97R05'  # Website ID in VNPAY System, get from config
	VNPAY_HASH_SECRET_KEY = 'GJNHOZCHXWLKBDDTXBNFDALKIZUXHCBA'  # Secret key for create checksum,get from config