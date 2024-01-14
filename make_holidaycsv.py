import pandas as pd
import ast
import requests

def get_holidays():
    url = 'https://www.nseindia.com/api/holiday-master?type=trading'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    page = requests.get(url, headers=headers)
    data = ast.literal_eval(page.text)
    df_holidays = pd.DataFrame(data['CBM'])

    return df_holidays

