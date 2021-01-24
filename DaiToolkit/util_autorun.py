import datetime
import os
import re
import subprocess
import sys
import time

import schedule
import win32com.client

from DaiToolkit import util_basics


class AutoRunner_Logger(object):
    """output to both terminal and log file"""

    def __init__(self):
        self.terminal = sys.stdout
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)

    def flush(self):
        pass


class AutoRunner(object):
    """ 
    auto runner of certain python script / script list 
    email in the class is for inner-team notification (includes error log)

    work_time in 24h format: "20:00" / "now"
    work_freq: day / hour / minute
    work_every: int
    runtype: "python" / "bat"
    """

    def __init__(self, py_path, work_every=1, work_freq="day", work_time_daily="08:00", emailtitle=None,
                 emailaddress=None, emailcc=None, runtype="python"):
        self.py_path_list = [py_path] if type(py_path) != list else py_path
        if work_time_daily == 'now':
            self.work_time_daily = util_basics.get_nextminute(minutes=0)
        elif re.match(r'\d\d:\d\d', work_time_daily):
            self.work_time_daily = work_time_daily
            self.work_now = False
        else:
            raise Exception("[work_time_daily] accept time like '18:03' or 'now'")

        self.work_every = work_every
        self.work_freq = work_freq
        self.emailaddress = emailaddress
        self.emailcc = emailcc
        self.emailtitle = self.py_path_list[0] if emailtitle is None else emailtitle
        self.runtype = runtype
        self.bat_path_list = []
        # if use bat run type/create bat file
        if runtype == "bat":
            self.batexecpy = util_basics.PYTHON_EXE_PATH
            self.activate = util_basics.ACTIVATE_BAT_PATH
            for py_path in self.py_path_list:
                bat_path = os.path.join(os.path.dirname(py_path), os.path.basename(py_path).replace(".py", ".bat"))
                self.bat_path_list.append(bat_path)

    def __send_email(self, body=""):
        """send email"""
        # ----------------------
        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)
        mail.To = self.emailaddress
        if self.emailcc:
            mail.CC = self.emailcc
        mail.Subject = self.emailtitle
        ## html style body message
        body_message = body
        # attach signature
        mail.GetInspector
        index = mail.HTMLbody.find('>', mail.HTMLbody.find('<body'))
        mail.HTMLbody = mail.HTMLbody[:index + 1] + body_message + mail.HTMLbody[index + 1:]
        # mail.Display(True)
        # Send the mail...
        mail.Send()
        print("Email sent successfully!")

    def pyfile_wrapper(self):
        # Output to log file
        default_stdout = sys.stdout
        mylog = AutoRunner_Logger()
        sys.stdout = mylog
        body_text = ""
        for num, py_path in enumerate(self.py_path_list):
            try:
                if self.runtype == "python":
                    exec(compile(open(py_path, "rb").read(), py_path, 'exec'))
                elif self.runtype == "bat":
                    with open(self.bat_path_list[num], 'w') as OPATH:
                        OPATH.writelines(['@echo off\n', os.path.abspath(self.activate) + "&&" + os.path.abspath(
                            self.batexecpy) + ' "' + os.path.abspath(py_path) + '"\n', 'exit'])
                    subprocess.call(self.bat_path_list[num], creationflags=subprocess.CREATE_NEW_CONSOLE, shell=False)
                else:
                    raise Exception("runtype error - support 'python' or 'bat'")
            except Exception as e:
                print(str(e))
                mylog.log = ["There is an issue running: ", py_path, "Error Msg:", str(e)]

            print("=====================================================")
            print("Finished Running [" + py_path + "]")
            print("=====================================================")
            body_text += "<br>".join(mylog.log)
        sys.stdout = default_stdout
        if self.emailaddress is not None:
            self.__send_email(body=body_text)

    def run(self):
        """will block the main thread"""
        if self.work_now:
            self.pyfile_wrapper()

        print("============== Auto Runner start working ============")
        if self.work_freq == "day":
            print("Working Frequency: Every " + str(
                self.work_every) + " " + self.work_freq + " at time " + self.work_time_daily)
            schedule.every(self.work_every).days.at(self.work_time_daily).do(self.pyfile_wrapper)
        elif self.work_freq == "hour":
            print("Working Frequency: Every " + str(self.work_every) + " " + self.work_freq)
            schedule.every(self.work_every).hours.do(self.pyfile_wrapper)
        elif self.work_freq == "minute":
            print("Working Frequency: Every " + str(self.work_every) + " " + self.work_freq)
            schedule.every(self.work_every).minutes.do(self.pyfile_wrapper)
        else:
            print("work_freq type error!")
            return

        while True:
            if self.work_freq == "day":
                print("Plan to Work at [" + self.work_time_daily + "] --- Time [" + datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S") + "]")
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    work_time_daily = util_basics.get_nextminute()
    py_path_list = []
    py_path_list.append("test1.py")
    py_path_list.append("test2.py")

    autorun_obj = AutoRunner(py_path_list, work_every=1, work_freq="day", work_time_daily=work_time_daily,
                             runtype="python")
    autorun_obj.run()
