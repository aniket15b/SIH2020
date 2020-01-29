
import datetime
from meraki_sdk.meraki_sdk_client import MerakiSdkClient
from meraki_sdk.exceptions.api_exception import APIException
from  meraki_sdk.controllers import mv_sense_controller
x_cisco_meraki_api_key = '15da0c6ffff295f16267f88f98694cf29a86ed87'

meraki = MerakiSdkClient(x_cisco_meraki_api_key)

collect = {}
def live(ser):
	collect['serial']=ser
	result = mv_sense_controller.get_device_camera_analytics_recent(collect)
	return result['entrances']
"""
from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html',result['entrances'])

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
"""