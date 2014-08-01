; Python defs and inits to be used by python calls within Csound instrumets


pyruni {{
import os
import csnd6
import random
import time

csInstance = csnd6.csoundGetInstance(_CSOUND_)
memoryPath = '../memoryRecording/'

def getBasenames(files):
    wavfiles = []
    txtfiles = []
    basenames = []
    # find wav and txt files
    for f in files:
        if (f.find('.wav') > -1):
            wavfiles.append(f)
        if (f.find('.txt') > -1):
            txtfiles.append(f)        
    # find base names for files that exist both with wav and txt extension
    for f in wavfiles:
        basename = f.split(".wav")[0]
        for t in txtfiles:
            if basename in t:
                basenames.append(basename)
    return basenames

def parseMarkerfile(basename):
    f = file(basename+".txt", 'r')
    markers = ''
    for line in f:
        try:
            num = float(line)
            markers += str(num)+' '
        except:
            pass    
        if "Total duration" in line:
            totaldur = float(line[15:])
    return markers, totaldur

ftableNums = []
# use ftable nums between 800 and 899, with markers loaded in tables 900 to 999
filesloaded ={}
# populate filesloaded with sublist [filename, ftablenum, accesstime]
# where accesstime is initially set to load time
# accesstime is updated each time Csound makes use of the file
# When Csound needs to use the file, query Python for the ftable num and update accesstime.
# GC: Files that has not been used for N minutes (20 min?) are GC'ed, 
# let Csound instrument to ftfree the table and also update Python list of available table nums

def getftableNum():
    tableNum = 800
    if len(ftableNums)<99:
        while tableNum in ftableNums:
            tableNum += 1
        if tableNum > 899: tableNum = 800
        ftableNums.append(tableNum)
    return tableNum

def loadAudioAndMarkers(basename):
    if basename not in filesloaded.keys():
        markers, totaldur = parseMarkerfile(memoryPath+basename)
        ftableNum = getftableNum()
        csnd6.csoundInputMessage(csInstance, 'i 71 0 .1 %i \"%s.wav\" \"%s\" %f'%(ftableNum, memoryPath+basename, markers, totaldur))
        print 'cs event sent:', 'i 71 0 .1 \"%s.wav\" \"%s\" %f'%(basename, markers, totaldur)
        filesloaded[basename] = [ftableNum, time.time()]
    else: print '%s already loaded'%basename

def soundfileGC(gcPeriod=20):
    print 'soundfileGC called'
    gcTime = time.time()-(gcPeriod*60)
    for f in filesloaded.keys():
        if filesloaded[f][1] < gcTime: 
            ftableNum = filesloaded[f][0]
            csnd6.csoundInputMessage(csInstance, 'i 72 0 .1 %i \"%s\"'%(ftableNum, f))
            print 'cs event sent:', 'i 72 0 .1 %i \"%s\"'%(ftableNum, f)

}}
