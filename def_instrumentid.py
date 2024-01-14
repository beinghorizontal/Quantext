import pandas as pd
import datetime
from def_expiry import expiry
import os


def getinstrumentid(symbol_name='NIFTY'):
    """

    :param symbol_name:
    :return: instrument token for the symbol name specified

    """

    root = 'g:/file/path/'
    datenow = datetime.datetime.now()+datetime.timedelta(3)
    year = datenow.strftime('%Y')
    month_num = datenow.strftime('%m')
    expiry_list = expiry(int(year))
    exp_day = expiry_list[int(month_num)]
    exp_int1 = datetime.datetime.strptime(exp_day, '%d-%m-%Y').date()
    exp_int = int(exp_int1.strftime('%Y%m%d'))
    now_int = int(datenow.strftime('%Y%m%d'))


    if now_int< exp_int:

        exp_day_format = datetime.datetime.strptime(exp_day, '%d-%m-%Y').date()
        expirydate = exp_day_format.strftime('%Y-%m-%d')
        print('expiry date ', expirydate)
    else:
        exp_day = expiry_list[int(month_num)+1]
        exp_day_format = datetime.datetime.strptime(exp_day, '%d-%m-%Y').date()
        expirydate = exp_day_format.strftime('%Y-%m-%d')
        print('expiry date rolled for next month', expirydate)

    #todo match file date with expiry. If higher then download instruments file
    mtime = os.path.getmtime(root+'data/instruments.csv')
    filedate1 = datetime.datetime.fromtimestamp(mtime).date()
    file_month = int(filedate1.strftime('%m'))
    exp_month = int(month_num)
    fileyear = int(filedate1.strftime('%Y'))

    """
        If zerodha's instruments.csv file is 2 months old or more than a year 
        we should download fresh copy because it stores
        only 3 months of instrument ids for Future contracts
        If it is fresh we will use local csv that we downloaded to save
        time because this file is a little big and takes few valuable seconds 
        to get downloaded.
        
    """
    if exp_month > file_month+1 or int(year) >fileyear:
        df1 = pd.read_csv('https://api.kite.trade/instruments')
        df1.to_csv(root+'data/instruments.csv',index=False)
    else:
        df1 = pd.read_csv(root+'data/instruments.csv')

    df1 = df1[df1['name'] == symbol_name]
    df1 = df1.sort_values('expiry', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    # exp_weekly = df1.iloc[0]['expiry']  # Uncomment for weekly expiry & comment below lines, I haven't tested recently
    df2 = df1[df1['expiry'] == expirydate]
    df2 = df2[df2.strike==0.0]

    zcodelist = [df2.iloc[0]['instrument_token']]

    return zcodelist
