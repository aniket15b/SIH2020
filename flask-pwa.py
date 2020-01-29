from flask import Flask, jsonify, render_template

from functions.live import live
from functions.number_of_students import view
from functions.past_records import past

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
@app.route('/detail')
def details():
    return render_template('detail.html')

@app.route('/offline.html')
def offline():
    return app.send_static_file('offline.html')


@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js')

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
