import time
import sys

from communication import send

secs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
mode = (sys.argv[2]) if len(sys.argv) >= 3 else 'both'

if mode == 'filelearn':
    send('playfile {}'.format(sys.argv[3]))
    print 'learning'
    send('startrec')
    time.sleep(3)
    send('stoprec')
    send('learn')
    print 'done learning'
elif mode == 'filerespond':
    send('playfile')
    print 'responding'
    send('startrec')
    time.sleep(3)
    send('stoprec')
    send('respond')
elif mode == 'both':
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
elif mode == 'learn':
    print 'learning'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
    print 'done learning'
elif mode == 'respond':
    print 'responding'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('respond')
else:
    print 'unknown mode', mode
    
#time.sleep(secs)
