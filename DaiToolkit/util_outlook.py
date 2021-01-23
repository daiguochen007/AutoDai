# -*- coding: utf-8 -*-

import re

import win32com.client


def outlook_tbl_bscfmt(df, index=False):
    """"""
    data_in_html = str(df.to_html(index=index))
    total_id = 'totalID'
    header_id = 'headerID'
    style_in_html = """<style>
                        table#{total_table} {{color='black';
                                              font-size:18px;
                                              border:0.2px solid black;
                                              border-collapse:collapse;
                                              table-layout:fixed;
                                              height="400";
                                              text-align:left;
                                              width:95%}}
                        thead#{header_table} {{background-color: 	#000000;
                                              color:#ffffff;
                                              }}
                        </style>""".format(total_table=total_id, header_table=header_id)
    data_in_html = re.sub(r'<table', r'<table id=%s ' % total_id, data_in_html)
    data_in_html = re.sub(r'<thead', r'<thead id=%s ' % header_id, data_in_html)
    data_in_html = re.sub(r' <tr style="text-align: right;">', ' <tr style="text-align: center;">', data_in_html)
    df_text = style_in_html + data_in_html
    return df_text


def send_email(title, body="", email_to='gd1023@nyu.edu', email_CC=None, att_list=[], add_signature=False):
    """ 
    Auto send email by outlook
    """
    try:
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To = email_to
        if email_CC:
            mail.CC = email_CC
        mail.Subject = title
        ## html style body message
        body_message = body
        # attach signature
        if add_signature:
            mail.GetInspector
            index = mail.HTMLbody.find('>', mail.HTMLbody.find('<body'))
            mail.HTMLbody = mail.HTMLbody[:index + 1] + body_message + mail.HTMLbody[index + 1:]
        mail.HTMLbody = body_message
        # mail.Display(True)
        # Send the mail...
        for att_path in att_list:
            mail.Attachments.Add(Source=att_path)
        mail.Send()
        print("Email sent successfully by Outlook!")
    except Exception as e:
        print("Fail sending Email:")
        print(str(e))
