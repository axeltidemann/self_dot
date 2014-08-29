#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Rule growth by employing the SPMF package.

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL

'''

from __future__ import print_function
import re
from subprocess import call

def convert(inputfile):
    '''Converts the text in the input file to the required format for the
    RuleGrowth implementation, where -1 indicates the end of an itemset
    and -2 indicates end of line. We treat *<name>*: as special indicators.'''

    txt = open(inputfile)
    symbols = []
    for line in txt.readlines():
        symbols.extend(line.split())
    unique = sorted(list(set(symbols)))

    name_re = re.compile(r'\**\*:') 
    txt = open(inputfile)
    outfile_name = inputfile + '-output'
    outfile = open(outfile_name, 'w')
    symbols = []
    name = []
    for line in txt.readlines():
        if name_re.findall(line):
            name = line.rstrip()
        elif line.strip():
            words = line.split()
            print(' '.join([ '{} {} -1'.format(unique.index(name), unique.index(w))
                             for w in words ]) + ' -2', 
                  file = outfile)
    
    return unique, outfile_name

def invert(inputfile, dictionary):
    '''Replaces the numbers in the output file with the corresponding words
    for human readibility.'''

    txt = open(inputfile)
    
    outfile = open(inputfile + '-inverted', 'w')
    
    for line in txt.readlines():
        line = line.strip()
        arrow = line.find('==>')
        pre = line[:arrow-1].split(',')
        sup = line.find('#SUP')
        post = line[arrow+4:sup-1].split(',')
        print('{} ==> {} {}'.format(','.join([ dictionary[int(p)] for p in pre]),
                                    ','.join([ dictionary[int(p)] for p in post]),
                                    line[sup:]),
              file = outfile)

if __name__ == '__main__':
    dictionary, outfile = convert('association_test_db_short.txt')
    rulefile = outfile + '-rules'
    call(['java', '-jar', 'spmf.jar', 'run', 'RuleGrowth', outfile, rulefile, '10\%', '50\%'])
    invert(rulefile, dictionary)

    
