import sys   #provides functions and variables which are used to manipulate different parts of the Python Runtime Environment
import logging
# creating a custom error message function
def error_message_detail(error,error_detail:sys):   #whenever an error-> push it to my own custom message. Error detail will be present inside sys
     _,_,exc_tb = error_detail.exc_info()            # file, line_no etc of exception
     file_name = exc_tb.tb_frame.f_code.co_filename # from documentation
     error_message = "Error occured in python script name [{0}], line number [{1}], error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
     )
     return error_message

class CustomException(Exception):                           # Exception is the parent class
    def __init__(self, error_message, error_detail:sys):    # constructor
        super().__init__(error_message)                       # inheritance- super.__init__ will execute __init__ method of Exception class
        self.error_message = error_message_detail(error_message,error_detail = error_detail)

    def __str__(self):
        return self.error_message




