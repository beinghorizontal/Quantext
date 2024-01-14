from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import pyotp


def kitelogin():
    """
        get credentials from local folder that you saved securely
        option to run Webdriver headless
        gte the zerodha url
        find xpath for use id, password and TOTP
        run potp to fetch op
        fetch and save tokens that we can use to download intraday data
        exit the chrome driver session

   """
    root = "Z:/kite/"
    file_names = ['kite_userid.txt', 'kite_pass.txt', 'kite_totp.txt']
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

    # options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1700, 1080)

    driver.get('https://kite.zerodha.com/')
    actionChain = webdriver.ActionChains(driver)
    actionChain.key_down(Keys.TAB).key_up(Keys.TAB).perform()

    driver.find_elements(By.XPATH,"//*[@type='text']")[0].send_keys(credentials[0])
    driver.find_elements(By.XPATH,"//*[@type='password']")[0].send_keys(credentials[1])  # //input[@id='password']
    driver.find_elements(By.XPATH,"//*[@type='submit']")[0].click()
    time.sleep(5)
    totp = pyotp.TOTP(credentials[2])
    otp = totp.now()
    ot = driver.find_elements(By.XPATH, "//input[@id='userid']")
    ot[0].send_keys(otp)
    time.sleep(22)

    driver.find_element(By.TAG_NAME,'body').send_keys(Keys.CONTROL + Keys.LEFT_SHIFT + '2')
    time.sleep(4)
    driver.find_element(By.TAG_NAME,'body').send_keys(Keys.ARROW_DOWN)
    time.sleep(1)
    driver.find_element(By.TAG_NAME,'body').send_keys('d')

    token_list = driver.get_cookies()
    try:
        token = token_list[0]['value']
        print(str(token_list),'\n','token=',token)

    except:
        print('toke failed')
        token = ''

    token_file = 'D:/anaconda/Scripts/token.txt'

    with open(token_file, 'w') as t:
        t.write('enctoken ' + token)
    time.sleep(6)
    driver.quit()
    # exit()

# sela()
