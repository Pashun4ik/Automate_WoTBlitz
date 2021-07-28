"""
'core.py' is the main control flow.
    - receives standard BGR input frames and arguments, then
    - executes instructions in return.
"""
import test
from Implementations import imageManipulation
from Implementations import controlOutput

class Automate(object):
    def __init__(self, input_frame, arguments):
        self.input_frame = input_frame
        self.arguments = arguments

        # Execute 'test.py' for development / debugging purposes when "--test" flag is passed
        # if "--test" not in arguments: killTest()
