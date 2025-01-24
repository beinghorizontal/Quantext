from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from eventOnDemand import event

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

def all_events():
    df = event()
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
    df = event()
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
    df = event()
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

def country_events(country = "India"):
    df = event()
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
    df = event()
    unique_countries = df['Country'].unique()
    unique_countries_str = ','.join(unique_countries)
    return unique_countries_str

with open('/home/pi/Scripts/telegram_credentials.txt', 'r') as f:
    credentials = f.read().splitlines()

token = credentials[0]
TELEGRAM_BOT_TOKEN = token

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "Hello! Welcome to the Quantext bot. Use /help to see the list of available commands.\n\n"
        "Don't forget to check out our YouTube channel for tutorials and more:\n"
        "https://www.youtube.com/@quantext"
    )
    await update.message.reply_text(welcome_message)

async def youtube(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    promo_message = (
        "Stay up-to-date with our latest tutorials and content on our YouTube channel:\n"
        "https://www.youtube.com/@quantext\n\n"
        "Subscribe and hit the bell icon to get notifications!"
    )
    await update.message.reply_text(promo_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('/start - Start the bot\n/help - Get help'
                                    '\n/events_all - Get all economic events'
                                    '\n/moderate_events - Get medium volatility economic events'
                                    '\n/high_vol_events - Get high volatility economic events'
                                    '\n/event_country - Get country specific economic events'
                                    '\n/country_list - Get country names in event calendar'
                                    '\n/youtube - Get YouTube channel link')


async def events_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    full_message = f'all_events: {all_events()}'
    max_length = 4000
    # Split the full message into smaller parts
    message_parts = split_message(full_message, max_length)
    # Send each part as a separate message
    for part in message_parts:
        await update.message.reply_text(part)

async def moderate_events(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    full_message = f'moderate_events: {moderate_volatility_events()}'
    max_length = 4000
    # Split the full message into smaller parts
    message_parts = split_message(full_message, max_length)
    # Send each part as a separate message
    for part in message_parts:
        await update.message.reply_text(part)


async def high_vol_events(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    full_message = f'high_volatility_events: {high_volatility_events()}'
    max_length = 4000
    # Split the full message into smaller parts
    message_parts = split_message(full_message, max_length)
    # Send each part as a separate message
    for part in message_parts:
        await update.message.reply_text(part)

async def country_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'country names: {get_country_names()}')

async def ct_events(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'country specific events: {country_events(country="India")}')

async def get_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    await update.message.reply_text(f'Current time: {now}')

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('events_all', events_all))
    app.add_handler(CommandHandler('moderate_events', moderate_events))
    app.add_handler(CommandHandler('high_vol_events', high_vol_events))
    app.add_handler(CommandHandler('country_list', country_list))
    app.add_handler(CommandHandler('event_country', ct_events))
    app.add_handler(CommandHandler('youtube', youtube))

    app.run_polling()

if __name__ == '__main__':
    main()
