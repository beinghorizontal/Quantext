RAlt::
Input Enterpad_key, L4 I T4
  if IsLabel(Enterpad_key)
    gosub %Enterpad_key%
return
zerp:
send, zerodha pass ; everything stores locally, someone can find password only through phisical access of your ahk script 
return
; runing command prompts replace your command inside quotes " your command"
fire:
run, %ComSpec% /c "scrcpy --tcpip=192.168.0.111"
;run, %ComSpec% /c "YourCommand"
pyth:
; replace path of your python script
run,g:/ anaconda/python G:\anaconda\Scripts\firestic_play.py
return
totp:
; it's cap insensitive
; from github repo download kiteotp file assuming you know how to get qr text 
; if you don't I explined n this video https://youtu.be/fRlfcR__XHU?si=ie_KQGN0epxDXKy6
run, g:/anaconda G:\anaconda\Scripts\kiteotp.py
return
