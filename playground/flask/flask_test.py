import os
from datetime import datetime

from flask import Flask, redirect, url_for, request, render_template, flash
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename

import DaiToolkit as tk

app = Flask(__name__)
app.secret_key = 'daviddaiissosmart'
app.config['UPLOAD_FOLDER'] = 'upload/'

login = tk.read_yaml(tk.PROJECT_CODE_PATH + "/DaiToolkit/login.yaml")["google"]
mail = Mail(app)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = login['user']
app.config['MAIL_PASSWORD'] = login['pass']
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


@app.route('/')
def main_page():
    my_int = 18
    my_str = 'curry'
    my_list = [1, 5, 4, 3, 2]
    my_dict = {
        'name': 'durant',
        'age': 28
    }

    # render_template方法:渲染模板
    # 参数1: 模板名称  参数n: 传到模板里的数据
    return render_template('main_page.html',
                           my_int=my_int,
                           my_str=my_str,
                           my_list=my_list,
                           my_dict=my_dict)


@app.route("/sendemail")
def index():
    msg = Message("This is an automsg from dai's portal [time = " + datetime.now().strftime("%H:%M:%S") + "]",
                  sender=login['user'], recipients=[login['user']])
    msg.body = "Hello Flask message sent from Flask-Mail"
    mail.send(msg)
    return "Sent"


@app.route('/p1')
def page_1():
    seccode_list = ['600383', '600585', '300750']
    price_dict = {}
    for k in seccode_list:
        price_dict[k] = float(tk.ts.get_realtime_quotes(k)["price"][0])
    return price_dict


@app.route('/hello/<name>/')
def hello_name(name):
    flash('You were successfully logged in! Dear ' + name)
    return redirect(url_for('page_1'))


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/login_res', methods=['POST', 'GET'])
def login_res():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return 'file uploaded successfully'


if __name__ == '__main__':
    # app.run()
    # host 127.0.0.1 as local host, port default 5000
    # host 0.0.0.0 to make it out availble (ipconfig -ipv4 -192.168.1.162)
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
