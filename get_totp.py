import pyotp
import pyperclip

"""
zerodha's QR text saved in this secured remote location
We will read that and put it in pyotp function to generate 
time based OTP
"""

root = "X:/totp_folder/"
with open(f'{root}totp.txt', 'r') as f:
    totp = f.read()
my_totp = pyotp.TOTP(totp)
got_totp = my_totp.now()
# print(otp)

# If you want to copy the totp number pip install pyperclip and use following line
# now totp will be in your clipboard
# Useful for non selenium based manual login without using phone for OTP
pyperclip.copy(got_totp)

