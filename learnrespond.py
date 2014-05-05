import time
import sys

from communication import send

secs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
mode = (sys.argv[2]) if len(sys.argv) >= 3 else 'both'

if mode == 'both':
    print 'learning'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
    print 'done learning'
    time.sleep(5)

print 'responding'
send('startrec')
time.sleep(secs)
send('stoprec')
send('respond')

#time.sleep(secs)
