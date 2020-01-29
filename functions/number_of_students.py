
import datetime
from meraki_sdk.meraki_sdk_client import MerakiSdkClient
from meraki_sdk.exceptions.api_exception import APIException
from  meraki_sdk.controllers import mv_sense_controller
x_cisco_meraki_api_key = '15da0c6ffff295f16267f88f98694cf29a86ed87'

meraki = MerakiSdkClient(x_cisco_meraki_api_key)

collect = {}
def view(ser,start,end):
	collect['serial'] = serial
	collect['t_1']=datetime.datetime.now()
	collect['t_0']=datetime.datetime.now() - datetime.timedelta(hours = 1)
	result =mv_sense_controller.get_device_camera_analytics_overview(collect)
	return result['entrances']

def check_number(num):
	if(num==view('serial1','2020-01-29 11:49:44.313206','2020-01-29 12:49:44.313206')):
		pass
	elif(num>view('serial1','2020-01-29 11:49:44.313206','2020-01-29 12:49:44.313206')):
		pass
	else:
		pass