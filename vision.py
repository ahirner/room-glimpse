from config import AZURE_COG_RETRIES, AZURE_COG_HOST
from creds.credentials import AZURE_COG_KEY

import requests

def processRequest(json, data, headers, params ):
    #From example code of project Oxford 
    """
    Parameters:
    json: Used when processing images from its URL. See API Documentation
    data: Used when processing image read from disk. See API Documentation
    headers: Used to pass the key information and the data type request
    """
    retries = 0
    result = None

    while True:
        response = requests.request( 'post', AZURE_COG_HOST, json = json, data = data, headers = headers, params = params )
        if response.status_code == 429: 
            print( "Message: %s" % ( response.json()['error']['message'] ) )
            if retries <= AZURE_COG_RETRIES: 
                time.sleep(1) 
                retries += 1
                continue
            else: 
                print( 'Error: failed after retrying!' )
                break

        elif response.status_code == 200 or response.status_code == 201:
            if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
                result = None 
            elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
                if 'application/json' in response.headers['content-type'].lower(): 
                    result = response.json() if response.content else None 
                elif 'image' in response.headers['content-type'].lower(): 
                    result = response.content
        else:
            print( "Error code: %d" % ( response.status_code ) )
            print( "Message: %s" % ( response.json()['error']['message'] ) )
        break
        
    return result

def analyze_img(jpg, features='Color,Categories,Tags,Description'):
    params = { 'visualFeatures' : features} 
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = AZURE_COG_KEY
    headers['Content-Type'] = 'application/octet-stream'
    result = processRequest(None, jpg, headers, params )
    
    return result