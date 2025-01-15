"""
 This bot scrapes the economic event calendar from Investing.com.

"""
import cloudscraper
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd

# Initialize cloudscraper
scraper = cloudscraper.create_scraper()

# URL for Investing.com economic calendar
url = 'https://www.investing.com/economic-calendar/'

def event():
    """
    This function fetches the economic event calendar from Investing.com, parses it,
    and returns a data frame of the events with the following columns:
    'Time', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous',
    'Volatility', 'Sentiment', 'Country'.
    """
    try:
        row_string_high = ""
        row_string_india = ""
        row_string_moderate = ""
        # Fetch the page content
        req = scraper.post(url)
        print(req.status_code)

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(req.text, 'html.parser')
        html_data = StringIO(soup.prettify())
        df = pd.read_html(html_data)
        df_ec = df[2]
        ev_date = df_ec.iloc[0, 1]
        print('Economic Event Calendar For:', ev_date)
        df_ec = df_ec.iloc[1:].reset_index(drop=True)

        # Find all economic event rows
        event_rows = soup.find_all('tr', class_='js-event-item')

        # Extract event details
        impacts = []
        countries = []
        for row in event_rows:
            impact_element = row.find('td', class_='left textNum sentiment noWrap')['title'].strip()
            country = row.find('td', class_='left flagCur noWrap').find('span', class_='ceFlags')['title'].strip()
            impacts.append(impact_element)
            countries.append(country)

        df_ec.columns = ['Time', 'Currency', 'Impact', 'Event', 'Actual', 'Forecast', 'Previous', 'Region', 'Volatility']
        df_ec = df_ec[df_ec['Impact'] != 'Holiday']
        df_ec = df_ec.reset_index()
        df_ec['Sentiment'] = impacts
        df_ec['Country'] = countries
        df_ec['date'] = ev_date
        df_ec_filter = df_ec[['date', 'Time', 'Country', 'Sentiment', 'Event', 'Actual', 'Forecast', 'Previous']]
        # Country-specific data frame
        return df_ec_filter
    except Exception as e:
        print(f"An error occurred: {e}")
        df_ec_filter = pd.DataFrame(columns=['date', 'Time', 'Country', 'Sentiment', 'Event', 'Actual', 'Forecast', 'Previous'])
        return df_ec_filter

