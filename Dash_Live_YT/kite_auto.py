#%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
import time
import pyotp
import urllib.parse as urlparse
#%%
print('imported libraries')
def kitelogin():
    """
        get credentials from local folder that you saved securely
        option to run Webdriver headless
        get the Zerodha kite login url
        find xpath for use id, password and TOTP
        fetch and save tokens that we can use to download intraday data
        exit the chrome driver session

   """
#%%
    root = "Z:/kite/"  # Replace with desire location
    file_names = ['kite_userid.txt', 'kite_pass.txt', 'kite_totp.txt']  # In above location make these 3 text files
    credentials = []
    for file in file_names:
        with open(f'{root}{file}', 'r') as f:
            content = f.read()
            credentials.append(content)

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

    headless = True
    options = Options()
    if headless:
            options.add_argument('--headless')
#%%
    # options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)
    # driver.set_window_size(1700, 1080)

    driver.get('https://kite.zerodha.com/')
    wait = WebDriverWait(driver, 20)

    # actionChain = webdriver.ActionChains(driver)
    # actionChain.key_down(Keys.TAB).key_up(Keys.TAB).perform()
    wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="text"]'))) \
        .send_keys(credentials[0])
    wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="password"]'))) \
        .send_keys(credentials[1])
    wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))) \
        .submit()
    totp = pyotp.TOTP(credentials[2])
    otp = totp.now()
    time.sleep(2)
    wait.until(EC.presence_of_element_located((By.XPATH, '//input[@id="userid"]'))) \
        .send_keys(otp)
#%%
    time.sleep(2)
    wait.until(EC.element_to_be_clickable((By.TAG_NAME, 'body'))) \
        .send_keys(Keys.CONTROL + Keys.LEFT_SHIFT + '2')

    token_list = driver.get_cookies()
    try:
        token = token_list[0]['value']
        print(str(token_list),'\n','token=',token)

    except:
        print('toke failed')
        token = ''
    #
    token_file = 'D:/anaconda/Scripts/token.txt'

    with open(token_file, 'w') as t:
        t.write('enctoken ' + token)
    time.sleep(1)
    driver.close()
    driver.quit()
# kitelogin()
