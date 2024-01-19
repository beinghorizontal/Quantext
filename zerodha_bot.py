import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import subprocess
import threading
import pyotp

root = "Z:/path/"
file_names = ['userid.txt', 'pass.txt', 'totp.txt']
credentials = []
for file in file_names:
    with open(f'{root}{file}', 'r') as f:
        content = f.read()
        credentials.append(content)

def run_chrome():
    subprocess.run("chrome.exe --remote-debugging-port=9240 --user-data-dir=D:/demos/selenium/chrome-win64")
    return
t1 = threading.Thread(target=run_chrome)
t1.start()

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument("--log-level=3")

options.add_experimental_option("debuggerAddress", "127.0.0.1:" + "9240")

bot = webdriver.Chrome(options=options)
bot.get('https://kite.zerodha.com/')
actionChain = webdriver.ActionChains(bot)
# With this method most like login id will get saved in cache so it will only ask for password
try:
    bot.find_elements(By.XPATH, "//*[@type='text']")[0].send_keys(credentials[0])
except:
    print('no login required')

xpathpass = "//input[@id='password']"
bot.find_element(By.XPATH, xpathpass).send_keys(credentials[1])
bot.find_elements(By.XPATH, "//*[@type='submit']")[0].click()
time.sleep(5)
totp = pyotp.TOTP(credentials[2])
otp = totp.now()
ot = bot.find_elements(By.XPATH, "//input[@id='userid']")
ot[0].send_keys(otp)
time.sleep(22)

xpath_risk = "//button[contains(text(),'I understand')]"
bot.find_element(By.XPATH,xpath_risk ).click()
"""
name should be exactly like you see in the watch window, it is cap sensitive
No need to say it should be in your watch list 1 
"""
scrip_name = 'GOLDBEES'
"""
xpath needs to be exactly like that, that's why single quote manipulation
"""
scrip_xpath = f"//span[contains(text(),"+"'"+scrip_name+"'"+")]"
bot.find_element(By.XPATH,scrip_xpath ).click()
# Replace "B" with "S" for selling
buy_path = "//button[contains(text(),'B')]"
bot.find_element(By.XPATH,buy_path).click()

"""
# The problem here is the xpath is dynamic and changes everytime
# so we have to crate dynamic xpath,it is less reliable than relative xpath but better 
# than absolute xpath
"""
xpath_dynamic = "//div[@class='no su-input-group su-static-label']//input[@type='number']"
# IF above method fails then use the below method, it's absolute xpath
# xpath_abs = "/html[1]/body[1]/div[1]/form[1]/section[1]/div[1]/div[2]/div[2]/div[1]/div[1]/div[1]/input[1]"
# bot.find_element(By.XPATH, xpath_abs).click()

bot.find_element(By.XPATH, xpath_dynamic).click()

bot.find_element(By.XPATH, xpath_dynamic).send_keys(Keys.BACKSPACE)
qunatity = '10'
bot.find_element(By.XPATH, xpath_dynamic).send_keys(qunatity)
"""
I don't want to create full fledged buy sell bot because you can't be
sure who else is going to use it and it is extremely dangerous affair
so I am leaving the actually buy and sell execution part.
If someone who understands the whole process and watched my youtube video
then that person won't have any issue finding xpath for buy and sell button.
This script suppose to treat as a test project because it is not fast as 
official API key by zerodha since using Selenium it elies on a lot of Java script which loads
concurrently and there is a danger that script gets executed before the java script fully loads 
result in failed execution.
Though most l;likely it won't buy instead of sell but watch manually nevertheless to 
see execution is working.
I have run similar script in the past and it works given you add ample amount of sleep time between
each pyton call.

"""
bot.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
# This wll close the active tab, if it's only 1 tab then will close browser
bot.close()
# It will shut down the browser and exit chromedriver instance
bot.quit()
# Close the thread
t1.join()
