from twilio.rest import Client
from datetime import datetime


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC08b30813340b59d07663c3180627c272'
auth_token = 'a5507bb530960d78e55e3f89121cf461'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="RBLBANK triggered"+ str(datetime.now()),
                     from_='+12018347323',
                     to='+919811244266'
                 )

print(message.sid)