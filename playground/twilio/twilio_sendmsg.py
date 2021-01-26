# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:21:04 2020

@author: Dai
"""

# send text message
from twilio.rest import Client

import DaiToolkit as tk

twilio_config = tk.read_yaml(tk.PROJECT_CODE_PATH + "/DaiToolkit/login.yaml")["twilio"]
account_sid = twilio_config['sid']
auth_token = twilio_config['auth_token']
client = Client(account_sid, auth_token)

#send to phone
message = client.messages.create(
    body="Hi, this is an twilio auto msg from Lao Dai",
    from_=twilio_config['phone_number'],
    to=twilio_config['my_cell'])

# print(message.sid)
