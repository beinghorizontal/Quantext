import requests
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from eventOnDemand import event

df = event()

def lowvolatility_events():
    row_string_all = ""
    if len(df) > 0:
        for index, row in df.iterrows():
            row_string_all = ', '.join([f"{col}: {row[col]}" for col in df.columns])
            # print(row_string_all)
    else:
        row_string_all = ""

    return row_string_all


def moderate_volatility_events():
    row_string_moderate = ""
    df_ec_moderate = df[df['Sentiment'].str.contains('Moderate Volatility Expected')]

    if len(df_ec_moderate) > 0:
        for index, row in df_ec_moderate.iterrows():
            row_string_moderate = ', '.join([f"{col}: {row[col]}" for col in df_ec_moderate.columns])
            # print(row_string_moderate)
    else:
        row_string_moderate = ""
        # print(row_string_moderate)

    return row_string_moderate


def high_volatility_events():
    row_string_high = ""
    df_ec_high = df[df['Sentiment'].str.contains('High Volatility Expected')]

    if len(df_ec_high) > 0:
        for index, row in df_ec_high.iterrows():
            row_string_high = ', '.join([f"{col}: {row[col]}" for col in df_ec_high.columns])
            # print(row_string_high)
    else:
        row_string_high = ""
        print(row_string_high)
    return row_string_high

def country_events(country = "India"):
    row_string_country = ""
    df_ec_country = df[df['Country'] == country]
    if len(df_ec_country) >0:
        for index, row in df_ec_country.iterrows():
            row_string_country = ', '.join([f"{col}: {row[col]}" for col in df_ec_country.columns])
            print(row_string_country)
    else:
        row_string_country = ""
        print(row_string_country)

    return row_string_country


def get_country_names():
    unique_countries = df['Country'].unique()
    unique_countries_str = ','.join(unique_countries)
    return unique_countries_str


with open('/home/pi/Scripts/telegram_credentials.txt', 'r') as f:
    credentials = f.read().splitlines()

TELEGRAM_BOT_TOKEN = credentials[0]
CHAT_ID = credentials[1]

def send_telegram_message(chat_id, message):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, json=payload)
    return response.json()


def main():
    # data = fetch_data()
    # Extract some information from the data to send in the message
    message0 = f"All Events:\n{lowvolatility_events()}"
    response0 = send_telegram_message(CHAT_ID, message0)

    message1 = f"High volatility Events:\n{high_volatility_events()}"
    response1 = send_telegram_message(CHAT_ID, message1)

    message2 = f"Moderate volatility Events:\n{moderate_volatility_events()}"
    response2 = send_telegram_message(CHAT_ID, message2)

    message3 = f"Country specific Events(India):\n{country_events()}"
    response3 = send_telegram_message(CHAT_ID, message3)

    print(response0, '\n', response0, '\n', response1, '\n', response2,
          '\n', response3)

if __name__ == '__main__':
    main()
