import datetime
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO,StringIO
# from def_get_fii import get_fii
# from tabulate import tabulate


# Convert the URL-encoded string to a regular string
# dayback=0
#del_url2 = "https://www.nseindia.com/api/reports?archives=name:CM - Security-wise Delivery Positions,type:archives,category:capital-market,section:equities&date=22-Aug-2019&type=equities&mode=single"

# dayback=0
def get_index(dayback=2):
    path = 'd:/demos/storage/hl'
    #cur_day = 0
    #prev_day = 1  # to download only current day use prev_day = 1
    date_list = []
    symbol_list = []
    adv_list = []
    dec_list = []
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    qty_list = []
    trades_list = []
    newhs = []
    newls = []
    # net_del_list = []

    # C FII
    fiiIdxFutlong_list = []
    fiiIdxFutshort_list = []
    fiiIdxCElong_list = []
    fiiIdxCEshort_list = []
    fiiIdxPElong_list = []
    fiiIdxPEshort_list = []

    fiiStkFutlong_list = []
    fiiStkFutshort_list = []
    fiiStkCElong_list = []
    fiiStkCEshort_list = []
    fiiStkPElong_list = []
    fiiStkPEshort_list = []

    # C DII
    diiIdxFutlong_list = []
    diiIdxFutshort_list = []
    diiIdxCElong_list = []
    diiIdxCEshort_list = []
    diiIdxPElong_list = []
    diiIdxPEshort_list = []

    diiStkFutlong_list = []
    diiStkFutshort_list = []
    diiStkCElong_list = []
    diiStkCEshort_list = []
    diiStkPElong_list = []
    diiStkPEshort_list = []

    # C Prop clients
    propIdxFutlong_list = []
    propIdxFutshort_list = []
    propIdxCElong_list = []
    propIdxCEshort_list = []
    propIdxPElong_list = []
    propIdxPEshort_list = []

    propStkFutlong_list = []
    propStkFutshort_list = []
    propStkCElong_list = []
    propStkCEshort_list = []
    propStkPElong_list = []
    propStkPEshort_list = []

    try:
        # dayback = i
        # dayback = 7
        dt_1 = datetime.date.today() - datetime.timedelta(dayback)  # 1 for y'day and 0 for today
        day_nse = dt_1.strftime("%d%m%y")
        day_del = dt_1.strftime("%d%m%Y")
        day_week = dt_1.strftime("%A")
        day_fii = dt_1.strftime("%d-%b-%Y")

        print(day_nse, day_week)

        # raw_url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA221223.csv"
        # raw_url_hl= "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR291223.zip"

        if day_week != 'Saturday' and day_week != 'Sunday':


            url = "https://nsearchives.nseindia.com/archives/equities/mkt/MA" + day_nse + ".csv"
            urlHL = "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR" + day_nse + ".zip"
            # delivery_url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day_del}.csv"
            # url_fii = f"https://nsearchives.nseindia.com/content/fo/fii_stats_{day_fii}.xls"
            url_part = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{day_del}.csv"
            # delivery_url2 = "https://www.nseindia.com/api/reports?archives=%5B%7B%22name%22%3A%22CM%20-%20Security-wise%20Delivery%20Positions%22%2C%22type%22%3A%22archives%22%2C%22category%22%3A%22capital-market%22%2C%22section%22%3A%22equities%22%7D%5D&date=01-Feb-2024&type=equities&mode=single"


            # fii_url = "https://www.nseindia.com/api/fiidiiTradeReact"  # C no historical data

            headers = {'Connection': 'keep-alive',
                       'authority': 'www.nseindia.com',
                       'path': '/api/marketStatus',
                       'Origin': 'https://www1.nseindia.com',
                       'Referer': 'https://www1.nseindia.com/products/content/equities/equities/archieve_eq.htm',
                       'Sec-Fetch-Mode': 'cors',
                       'Sec-Fetch-Site': 'same-origin',
                       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

            req = requests.get(url, headers=headers)
            response = requests.get(urlHL, headers=headers)
            # req_del = requests.get(delivery_url, headers=headers)
            #req_fii = requests.get(url_fii, headers=headers)
            req_part = requests.get(url_part, headers=headers)
            # req_del2 = requests.get(delivery_url2,headers=headers)
            # print(req_del2.status_code)
            if req.status_code == 200 and response.status_code==200 and req_part.status_code == 200:
                with ZipFile(BytesIO(response.content)) as zip_file:
                    # Extract all files in the zip to a temporary directory
                    zip_file.extractall(path)
                csv_path = path + '/HL' + day_nse + '.csv'
                dfHL = pd.read_csv(csv_path)
                dfHL['numeric'] = np.where(dfHL['NEW_STATUS'] == 'H', 1, 0)
                newh = dfHL['numeric'].sum()
                newl = len(dfHL) - newh

                # data = StringIO(req_del.text)
                # df = pd.read_csv(data)
                # df_eq = df[df[' SERIES'].str.contains("EQ")]
                # df_eq2 = df_eq[df_eq[' TURNOVER_LACS'].astype(int) > 10000]
                # net_del = df_eq2[' DELIV_PER'].astype(float).sum()/len(df_eq2)

                list_date = req.text.split('\n')
                df = pd.DataFrame(list_date)

                date_obj = pd.to_datetime(dt_1, format="%d%m-%Y")
                dfsplit = df[0].str.split(',', expand=True)
                adv_row = dfsplit.index[dfsplit[1] == 'ADVANCES'].tolist()
                adv_row = adv_row[0]
                adv = float(dfsplit.iloc[adv_row][2])
                dec = float(dfsplit.iloc[adv_row+1][2])
                try:

                    nfrow = dfsplit.index[dfsplit[1]=='Nifty 50'].tolist()
                    nfrow = nfrow[0]
                except:
                    nfrow = dfsplit.index[dfsplit[1] == 'CNX Nifty'].tolist()
                    nfrow = nfrow[0]


                # try:
                #     adv_str = df.iloc[adv_row, 0]
                #     dec_str = df.iloc[adv_row+1, 0]
                #     adv_r = adv_str.split(",")[2]
                #     dec_r = dec_str.split(",")[2]
                #     adv = float(adv_r)
                #     dec = float(dec_r)
                # except:
                #     adv_str = df.iloc[60, 0]
                #     dec_str = df.iloc[61, 0]
                #     adv_r = adv_str.split(",")[2]
                #     dec_r = dec_str.split(",")[2]
                #     adv = float(adv_r)
                #     dec = float(dec_r)
                # nifty_val = dfsplit.iloc[nfrow][1]
                # nifty_list = str(nifty_val).split(',')

                sym = dfsplit.iloc[nfrow][1].strip()
                o = dfsplit.iloc[nfrow][3].strip()  # float(nifty_list[3].strip())
                h = dfsplit.iloc[nfrow][4].strip()
                l = dfsplit.iloc[nfrow][5].strip()
                c = dfsplit.iloc[nfrow][6].strip()
                value_row =  dfsplit.index[dfsplit[1] == ' Traded Value (Rs. In Crores)'].tolist()
                qty = float(dfsplit.iloc[value_row[0]+1][2])
                trades = float(dfsplit.iloc[value_row[0]+2][2])
                text = dfsplit.iloc[1][0]

                path = 'd:/demos/output.xls'
                with open(path, 'wb') as f:
                    f.write(req_part.content)
                # C participation wise OI data
                data_part = StringIO(req_part.text)
                dfPart = pd.read_csv(data_part)
                dfPart.columns = dfPart.iloc[0]
                dfPart = dfPart.iloc[1:]
                last_column_number = dfPart.shape[1] - 1  # Get the column number of the last column
                dfPart = dfPart.drop(dfPart.columns[last_column_number], axis=1)

                # C FII data
                # print(dfPart.columns.tolist)

                # C FII ..........................................**************************..................
                fii_row = dfPart[dfPart['Client Type'].str.contains('FII')].index[0]-1
                # print(custom_tabulate(dfPart))

                print('Client Type: ', dfPart.iloc[fii_row]['Client Type'])
                # fiiIdxFutlong = dfPart.iloc[fii_row]['Future Index Long']
                # fiiIdxFutshort = dfPart.iloc[fii_row]['Future Index Short']
                # fiiIdxCElong = dfPart.iloc[fii_row]['Option Index Call Long']
                # fiiIdxCEshort = dfPart.iloc[fii_row]['Option Index Call Short']
                # fiiIdxPElong = dfPart.iloc[fii_row]['Option Index Put Long']
                # fiiIdxPEshort = dfPart.iloc[fii_row]['Option Index Put Short']
                #
                # fiiStkFutlong = dfPart.iloc[fii_row]['Future Stock Long']
                # fiiStkFutshort = dfPart.iloc[fii_row]['Future Stock Short\t']
                # fiiStkCElong = dfPart.iloc[fii_row]['Option Stock Call Long']
                # fiiStkCEshort = dfPart.iloc[fii_row]['Option Stock Call Short']
                # fiiStkPElong = dfPart.iloc[fii_row]['Option Stock Put Long']
                # fiiStkPEshort = dfPart.iloc[fii_row]['Option Stock Put Short']

                # C append values in list
                fiiIdxFutlong_list.append(dfPart.iloc[fii_row]['Future Index Long'])
                fiiIdxFutshort_list.append(dfPart.iloc[fii_row]['Future Index Short'])
                fiiIdxCElong_list.append(dfPart.iloc[fii_row]['Option Index Call Long'])
                fiiIdxCEshort_list.append(dfPart.iloc[fii_row]['Option Index Call Short'])
                fiiIdxPElong_list.append(dfPart.iloc[fii_row]['Option Index Put Long'])
                fiiIdxPEshort_list.append(dfPart.iloc[fii_row]['Option Index Put Short'])

                fiiStkFutlong_list.append(dfPart.iloc[fii_row]['Future Stock Long'])
                # print(dfPart.columns.tolist)
                try:
                    fiiStkFutshort_list.append(dfPart.iloc[fii_row]['Future Stock Short\t'])
                except:
                    try:
                        fiiStkFutshort_list.append(dfPart.iloc[fii_row]['Future Stock Short'])
                    except:
                        fiiStkFutshort_list.append(dfPart.iloc[fii_row]['Future Stock Short       '])

                fiiStkCElong_list.append(dfPart.iloc[fii_row]['Option Stock Call Long'])
                fiiStkCEshort_list.append(dfPart.iloc[fii_row]['Option Stock Call Short'])
                fiiStkPElong_list.append(dfPart.iloc[fii_row]['Option Stock Put Long'])
                fiiStkPEshort_list.append(dfPart.iloc[fii_row]['Option Stock Put Short'])

                # C DII ............................... ********************************......................

                dii_row = dfPart[dfPart['Client Type'].str.contains('DII')].index[0] - 1
                print('Client Type: ', dfPart.iloc[dii_row]['Client Type'])
                # diiIdxFutlong = dfPart.iloc[dii_row]['Future Index Long']
                # diiIdxFutshort = dfPart.iloc[dii_row]['Future Index Short']
                # diiIdxCElong = dfPart.iloc[dii_row]['Option Index Call Long']
                # diiIdxCEshort = dfPart.iloc[dii_row]['Option Index Call Short']
                # diiIdxPElong = dfPart.iloc[dii_row]['Option Index Put Long']
                # diiIdxPEshort = dfPart.iloc[dii_row]['Option Index Put Short']
                #
                # diiStkFutlong = dfPart.iloc[dii_row]['Future Stock Long']
                # diiStkFutshort = dfPart.iloc[dii_row]['Future Stock Short\t']
                # diiStkCElong = dfPart.iloc[dii_row]['Option Stock Call Long']
                # diiStkCEshort = dfPart.iloc[dii_row]['Option Stock Call Short']
                # diiStkPElong = dfPart.iloc[dii_row]['Option Stock Put Long']
                # diiStkPEshort = dfPart.iloc[dii_row]['Option Stock Put Short']

                # C Prop ............................... ********************************......................
                diiIdxFutlong_list.append(dfPart.iloc[dii_row]['Future Index Long'])
                diiIdxFutshort_list.append(dfPart.iloc[dii_row]['Future Index Short'])
                diiIdxCElong_list.append(dfPart.iloc[dii_row]['Option Index Call Long'])
                diiIdxCEshort_list.append(dfPart.iloc[dii_row]['Option Index Call Short'])
                diiIdxPElong_list.append(dfPart.iloc[dii_row]['Option Index Put Long'])
                diiIdxPEshort_list.append(dfPart.iloc[dii_row]['Option Index Put Short'])

                diiStkFutlong_list.append(dfPart.iloc[dii_row]['Future Stock Long'])
                try:
                    diiStkFutshort_list.append(dfPart.iloc[dii_row]['Future Stock Short\t'])
                except:
                    try:
                        diiStkFutshort_list.append(dfPart.iloc[dii_row]['Future Stock Short'])
                    except:
                        diiStkFutshort_list.append(dfPart.iloc[dii_row]['Future Stock Short       '])

                diiStkCElong_list.append(dfPart.iloc[dii_row]['Option Stock Call Long'])
                diiStkCEshort_list.append(dfPart.iloc[dii_row]['Option Stock Call Short'])
                diiStkPElong_list.append(dfPart.iloc[dii_row]['Option Stock Put Long'])
                diiStkPEshort_list.append(dfPart.iloc[dii_row]['Option Stock Put Short'])

                prop_row = dfPart[dfPart['Client Type'].str.contains('Pro')].index[0] - 1
                print('Client Type: ', dfPart.iloc[prop_row]['Client Type'])
                # propIdxFutlong = dfPart.iloc[prop_row]['Future Index Long']
                # propIdxFutshort = dfPart.iloc[prop_row]['Future Index Short']
                # propIdxCElong = dfPart.iloc[prop_row]['Option Index Call Long']
                # propIdxCEshort = dfPart.iloc[prop_row]['Option Index Call Short']
                # propIdxPElong = dfPart.iloc[prop_row]['Option Index Put Long']
                # propIdxPEshort = dfPart.iloc[prop_row]['Option Index Put Short']
                #
                # propStkFutlong = dfPart.iloc[prop_row]['Future Stock Long']
                # propStkFutshort = dfPart.iloc[prop_row]['Future Stock Short\t']
                # propStkCElong = dfPart.iloc[prop_row]['Option Stock Call Long']
                # propStkCEshort = dfPart.iloc[prop_row]['Option Stock Call Short']
                # propStkPElong = dfPart.iloc[prop_row]['Option Stock Put Long']
                # propStkPEshort = dfPart.iloc[prop_row]['Option Stock Put Short']
                # print(f"propIdxFutlong: {propIdxFutlong}")
                # print(f"propIdxFutshort: {propIdxFutshort}")
                # print(f"propIdxCElong: {propIdxCElong}")
                # print(f"propIdxCEshort: {propIdxCEshort}")
                # print(f"propIdxPElong: {propIdxPElong}")
                # print(f"propIdxPEshort: {propIdxPEshort}")
                #
                # print(f"propStkFutlong: {propStkFutlong}")
                # print(f"propStkFutshort: {propStkFutshort}")
                # print(f"propStkCElong: {propStkCElong}")
                # print(f"propStkCEshort: {propStkCEshort}")
                # print(f"propStkPElong: {propStkPElong}")
                # print(f"propStkPEshort: {propStkPEshort}")

                propIdxFutlong_list.append(dfPart.iloc[prop_row]['Future Index Long'])
                propIdxFutshort_list.append(dfPart.iloc[prop_row]['Future Index Short'])
                propIdxCElong_list.append(dfPart.iloc[prop_row]['Option Index Call Long'])
                propIdxCEshort_list.append(dfPart.iloc[prop_row]['Option Index Call Short'])
                propIdxPElong_list.append(dfPart.iloc[prop_row]['Option Index Put Long'])
                propIdxPEshort_list.append(dfPart.iloc[prop_row]['Option Index Put Short'])

                propStkFutlong_list.append(dfPart.iloc[prop_row]['Future Stock Long'])
                try:
                    propStkFutshort_list.append(dfPart.iloc[prop_row]['Future Stock Short\t'])
                except:
                    try:
                        propStkFutshort_list.append(dfPart.iloc[prop_row]['Future Stock Short'])
                    except:
                        propStkFutshort_list.append(dfPart.iloc[prop_row]['Future Stock Short       '])

                propStkCElong_list.append(dfPart.iloc[prop_row]['Option Stock Call Long'])
                propStkCEshort_list.append(dfPart.iloc[prop_row]['Option Stock Call Short'])
                propStkPElong_list.append(dfPart.iloc[prop_row]['Option Stock Put Long'])
                propStkPEshort_list.append(dfPart.iloc[prop_row]['Option Stock Put Short'])

                date_list.append(date_obj)



                # print(custom_tabulate(df_combined))


                # net_amount = float(dfPart.iloc[indexf_r]['buy_amount']) - float(dfPart.iloc[indexf_r]['sell_amount'])
                # C Options Index
                # indexo_r = dfPart[dfPart['names'].str.contains('INDEX OPTIONS')].index[0]-1
                # options_oi_contracts = dfPart.iloc[indexo_r]['oi']
                # options__oi_value = dfPart.iloc[indexo_r]['oi_value']
                # options__net_amount = float(dfPart.iloc[indexo_r]['buy_amount']) - float(dfPart.iloc[indexo_r]['sell_amount'])
                # # C Stocks Futures
                # stockf_r = dfPart[dfPart['names'].str.contains('STOCK FUTURES')].index[0]-1
                # stocks_oi_contracts = dfPart.iloc[stockf_r]['oi']
                # stocks__oi_value = dfPart.iloc[stockf_r]['oi_value']
                # stocks__net_amount = float(dfPart.iloc[stockf_r]['buy_amount']) - float(dfPart.iloc[stockf_r]['sell_amount'])
                # # C Stocks Options
                # stocko_r = dfPart[dfPart['names'].str.contains('STOCK OPTIONS')].index[0]-1
                # options_stocks_oi_contracts = dfPart.iloc[stocko_r]['oi']
                # options_stocks__oi_value = dfPart.iloc[stocko_r]['oi_value']
                # options_stocks__net_amount = float(dfPart.iloc[stocko_r]['buy_amount']) - float(
                #     dfPart.iloc[stocko_r]['sell_amount'])

                # append to lists
                newhs.append(newh)
                newls.append(newl)
                symbol_list.append(sym)
                open_list.append(o)
                high_list.append(h)
                low_list.append(l)
                close_list.append(c)
                qty_list.append(qty)
                trades_list.append(trades)
                adv_list.append(int(adv))
                dec_list.append(int(dec))
                # net_del_list.append(round(net_del,2))


            else:
                print('connection error or Holiday')
                text = ''
        else:
            print('weekend', day_week)
            text = ''

    except Exception as e:
        print(e)
        text = ''

    df_combined = pd.DataFrame({
        'date': date_list,
        'fiiIdxFutlong': fiiIdxFutlong_list,
        'fiiIdxFutshort': fiiIdxFutshort_list,
        'fiiIdxCElong': fiiIdxCElong_list,
        'fiiIdxCEshort': fiiIdxCEshort_list,
        'fiiIdxPElong': fiiIdxPElong_list,
        'fiiIdxPEshort': fiiIdxPEshort_list,
        'fiiStkFutlong': fiiStkFutlong_list,
        'fiiStkFutshort': fiiStkFutshort_list,
        'fiiStkCElong': fiiStkCElong_list,
        'fiiStkCEshort': fiiStkCEshort_list,
        'fiiStkPElong': fiiStkPElong_list,
        'fiiStkPEshort': fiiStkPEshort_list,
        'diiIdxFutlong': diiIdxFutlong_list,
        'diiIdxFutshort': diiIdxFutshort_list,
        'diiIdxCElong': diiIdxCElong_list,
        'diiIdxCEshort': diiIdxCEshort_list,
        'diiIdxPElong': diiIdxPElong_list,
        'diiIdxPEshort': diiIdxPEshort_list,
        'diiStkFutlong': diiStkFutlong_list,
        'diiStkFutshort': diiStkFutshort_list,
        'diiStkCElong': diiStkCElong_list,
        'diiStkCEshort': diiStkCEshort_list,
        'diiStkPElong': diiStkPElong_list,
        'diiStkPEshort': diiStkPEshort_list,
        'propIdxFutlong': propIdxFutlong_list,
        'propIdxFutshort': propIdxFutshort_list,
        'propIdxCElong': propIdxCElong_list,
        'propIdxCEshort': propIdxCEshort_list,
        'propIdxPElong': propIdxPElong_list,
        'propIdxPEshort': propIdxPEshort_list,
        'propStkFutlong': propStkFutlong_list,
        'propStkFutshort': propStkFutshort_list,
        'propStkCElong': propStkCElong_list,
        'propStkCEshort': propStkCEshort_list,
        'propStkPElong': propStkPElong_list,
        'propStkPEshort': propStkPEshort_list,
    })

    df1 = pd.DataFrame({'date': date_list, 'symbol': symbol_list,
                        'adv': adv_list,
                        'dec': dec_list, 'open': open_list,
                        'high': high_list, 'low': low_list,
                        'close': close_list, 'qty': qty_list,
                        'trades': trades_list,'new_high':newhs, 'new_low':newls,
                        'fiiIdxFutlong': fiiIdxFutlong_list,
                        'fiiIdxFutshort': fiiIdxFutshort_list,
                        'fiiIdxCElong': fiiIdxCElong_list,
                        'fiiIdxCEshort': fiiIdxCEshort_list,
                        'fiiIdxPElong': fiiIdxPElong_list,
                        'fiiIdxPEshort': fiiIdxPEshort_list,
                        'fiiStkFutlong': fiiStkFutlong_list,
                        'fiiStkFutshort': fiiStkFutshort_list,
                        'fiiStkCElong': fiiStkCElong_list,
                        'fiiStkCEshort': fiiStkCEshort_list,
                        'fiiStkPElong': fiiStkPElong_list,
                        'fiiStkPEshort': fiiStkPEshort_list,
                        'diiIdxFutlong': diiIdxFutlong_list,
                        'diiIdxFutshort': diiIdxFutshort_list,
                        'diiIdxCElong': diiIdxCElong_list,
                        'diiIdxCEshort': diiIdxCEshort_list,
                        'diiIdxPElong': diiIdxPElong_list,
                        'diiIdxPEshort': diiIdxPEshort_list,
                        'diiStkFutlong': diiStkFutlong_list,
                        'diiStkFutshort': diiStkFutshort_list,
                        'diiStkCElong': diiStkCElong_list,
                        'diiStkCEshort': diiStkCEshort_list,
                        'diiStkPElong': diiStkPElong_list,
                        'diiStkPEshort': diiStkPEshort_list,
                        'propIdxFutlong': propIdxFutlong_list,
                        'propIdxFutshort': propIdxFutshort_list,
                        'propIdxCElong': propIdxCElong_list,
                        'propIdxCEshort': propIdxCEshort_list,
                        'propIdxPElong': propIdxPElong_list,
                        'propIdxPEshort': propIdxPEshort_list,
                        'propStkFutlong': propStkFutlong_list,
                        'propStkFutshort': propStkFutshort_list,
                        'propStkCElong': propStkCElong_list,
                        'propStkCEshort': propStkCEshort_list,
                        'propStkPElong': propStkPElong_list,
                        'propStkPEshort': propStkPEshort_list,
                        })

    # df1 = df1.set_index('date')
    df2 = df1.sort_index(ascending=False)
    return df2, df_combined




