import requests
from telegram import Bot
from telegram.ext import Updater, CommandHandler
from eventOnDemand import event

df = event()


def lowvolatility_events():
    row_string_all = ""
    string_all = ""
    if len(df) > 0:
        for index, row in df.iterrows():
            row_string_all = ', '.join([f"{col}: {row[col]}" for col in df.columns])
            string_all += '\n' + row_string_all
            # print(row_string_all)
    else:
        row_string_all = ""
        string_all += '\n' + row_string_all

    return string_all


def moderate_volatility_events():
    row_string_moderate = ""
    string_all_moderate = ""
    df_ec_moderate = df[df['Sentiment'].str.contains('Moderate Volatility Expected')]

    if len(df_ec_moderate) > 0:
        for index, row in df_ec_moderate.iterrows():
            row_string_moderate = ', '.join([f"{col}: {row[col]}" for col in df_ec_moderate.columns])
            string_all_moderate += '\n' + row_string_moderate
            # print(row_string_moderate)
    else:
        row_string_moderate = ""
        string_all_moderate += '\n' + row_string_moderate
        # print(row_string_moderate)

    return string_all_moderate


def high_volatility_events():
    row_string_high = ""
    string_all_high = ""
    df_ec_high = df[df['Sentiment'].str.contains('High Volatility Expected')]
    string_all_high += '\n' + row_string_high

    if len(df_ec_high) > 0:
        for index, row in df_ec_high.iterrows():
            row_string_high = ', '.join([f"{col}: {row[col]}" for col in df_ec_high.columns])
            string_all_high += '\n' + row_string_high

            # print(row_string_high)
    else:
        row_string_high = ""
        string_all_high += '\n' + row_string_high
    return string_all_high


def country_events(country="India"):
    row_string_country = ""
    string_all_country = ""

    df_ec_country = df[df['Country'] == country]
    if len(df_ec_country) > 0:
        for index, row in df_ec_country.iterrows():
            row_string_country = ', '.join([f"{col}: {row[col]}" for col in df_ec_country.columns])
            string_all_country += '\n' + row_string_country

            # print(row_string_country)
    else:
        row_string_country = ""
        string_all_country += '\n' + row_string_country
        # print(row_string_country)
    return string_all_country


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

# Function to split a long message into multiple messages. Telegram has a limit of 4096 characters per message
def split_message(message, max_length):
    parts = []
    while len(message) > max_length:
        part = message[:max_length]
        last_space = part.rfind(' ')
        if last_space == -1:
            last_space = max_length
        parts.append(message[:last_space])
        message = message[last_space:].strip()
    parts.append(message)
    return parts


def main():
    # data = fetch_data()
    # Extract some information from the data to send in the message
    message0 = f"All Events:\n{lowvolatility_events()}"
    if len(message0) > 4000:
        split_messages = split_message(message0, 4000)
        for i, msg in enumerate(split_messages):
            send_telegram_message(CHAT_ID, f"{i+1}:\n{msg}\n")
    else:
        response0 = send_telegram_message(CHAT_ID, message0)

    message1 = f"High volatility Events:\n{high_volatility_events()}"
    if len(message1) > 4000:
        split_messages = split_message(message1, 4000)
        for i, msg in enumerate(split_messages):
            send_telegram_message(CHAT_ID, f"{i+1}:\n{msg}\n")
    else:
        response1 = send_telegram_message(CHAT_ID, message1)

    message2 = f"Moderate volatility Events:\n{moderate_volatility_events()}"

    if len(message2) > 4000:
        split_messages = split_message(message2, 4000)
        for i, msg in enumerate(split_messages):
            send_telegram_message(CHAT_ID, f"{i+1}:\n{msg}\n")
    else:
        response2 = send_telegram_message(CHAT_ID, message2)

    message3 = f"Country specific Events(India):\n{country_events()}"

    if len(message3) > 4000:
        split_messages = split_message(message3, 4000)
        for i, msg in enumerate(split_messages):
            send_telegram_message(CHAT_ID, f"{i+1}:\n{msg}\n")
    else:
        response3 = send_telegram_message(CHAT_ID, message3)



if __name__ == '__main__':
    main()
