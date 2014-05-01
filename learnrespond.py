import time
import sys

from communication import send

secs = int(sys.argv[1]) if len(sys.argv) == 2 else 5

send('startrec')
time.sleep(secs)
send('stoprec')
send('learn')

time.sleep(5)

send('startrec')
time.sleep(secs)
send('stoprec')
send('respond')

time.sleep(secs)
