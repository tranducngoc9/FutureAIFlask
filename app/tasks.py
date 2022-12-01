from flask import jsonify, json
from pprint import pprint
import requests

def post_processing(url, user_dict, type):

    if user_dict["user_data"]["phone"]:
        r = requests.post('http://zalo.ngochip.net/api/v1/zalo', json={"phone":user_dict["user_data"]["phone"], "message":"Xin chào " + user_dict["full_name"]})
    
    #r = requests.post('http://zalo.ngochip.net/api/v1/zalo', json={"phone":"0925058088", "message":"Xin chào "})
    r = requests.post(url, json={"event":type, "data": json.dumps(user_dict)})
    pprint(vars(r)) 
    return 1
