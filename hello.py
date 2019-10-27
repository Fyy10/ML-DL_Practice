'This is a hello world program.'

__author__ = 'Jeff Fu'

import sys

def output():
	args = sys.argv
	if len(args) == 1:
		print 'Hello World!'
	elif len(args) == 2:
		print 'Hello', args[1] + '!'
	else:
		print 'Too many arguments'
if __name__ == '__main__':
	output()
