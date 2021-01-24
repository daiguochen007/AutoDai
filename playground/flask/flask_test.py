from flask import Flask

import DaiToolkit as tk

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'This is a test for David self-use flask app - everything will be api in the future [thumbs up]'


@app.route('/p1')
def page_1():
    seccode_list = ['600383', '600585', '300750']
    price_dict = {}
    for k in seccode_list:
        price_dict[k] = float(tk.ts.get_realtime_quotes(k)["price"][0])
    return price_dict


@app.route('/hello/<name>')
def hello_name(name):
    return 'Hello %s!' % name


if __name__ == '__main__':
    # app.run()
    # host 127.0.0.1 as local host, port default 5000
    # host 0.0.0.0 to make it out availble (ipconfig -ipv4 -192.168.1.162)
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
