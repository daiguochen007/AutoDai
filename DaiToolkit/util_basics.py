# -*- coding: utf-8 -*-

import datetime
import inspect
import os
import shutil
import subprocess
import sys
import time
from functools import wraps
from types import FunctionType

import numpy as np
from pympler import asizeof

###########################################################
#  Global Variables
###########################################################
PYTHON_EXE_PATH = r'C:\ProgramData\Anaconda2\envs\py3\python.exe'
ACTIVATE_BAT_PATH = os.path.join(os.path.dirname(PYTHON_EXE_PATH), 'Scripts', 'activate.bat')

PROJECT_ROOT_PATH = "C:/Users/Dai/Desktop/investment"
PROJECT_CODE_PATH = PROJECT_ROOT_PATH + "/Git/AutoDai_py3"
PROJECT_DATA_PATH = PROJECT_ROOT_PATH + "/data"


###########################################################
#  Funcs
###########################################################
def install_package(pkg_name, install_type='easy_install', version=None):
    print("================ Start Install Package [" + pkg_name + "] ================")
    if install_type == "easy_install":
        print("*** easy_install ***")
        res = subprocess.check_output(["easy_install", pkg_name])
    elif install_type == "pip":
        print("*** pip install ***")
        if version is not None:
            res = subprocess.check_output(
                ["pip", "install", "--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org",
                 pkg_name + "==" + version])
        else:
            res = subprocess.check_output(
                ["pip", "install", "--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org", pkg_name])
    else:
        raise Exception("install_type error, support 'pip' or 'easy_install'")
    for line in res.splitlines():
        print(line)


def kill_process(process_exe_name):
    res = os.system("taskkill /F /IM " + process_exe_name + " /T")
    if res == 0:
        print("Successfully kill [" + process_exe_name + "] process")
    else:
        print("Fail to kill, no [" + process_exe_name + "] process")


def get_obj_size(pyobject, size_type="ALL"):
    """
    get size of given python object
    """
    if size_type == "ALL":
        num_bytes = asizeof.asizeof(pyobject)
    else:
        num_bytes = float(sys.getsizeof(pyobject))

    if num_bytes < 1024:
        print(str(round(num_bytes, 2)) + " B")
    elif num_bytes < 1024 * 1024:
        print(str(round(num_bytes / 1024, 2)) + " KB")
    elif num_bytes < 1024 * 1024 * 1024:
        print(str(round(num_bytes / (1024 * 1024), 2)) + " MB")
    else:
        print(str(round(num_bytes / (1024 * 1024 * 1024), 2)) + " GB")


def decor_runtime(some_function):
    """
    Outputs the time a function takes to execute.
    """

    @wraps(some_function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        try:
            print("========== Running Function [" + some_function.__name__ + "] ==========")
            print("=== [" + some_function.__name__ + "] stdout:")
            return some_function(*args, **kwargs)
        finally:
            t2 = time.time()
            print("=== [" + some_function.__name__ + "] Time usage: " + time_interval_from_secs(t2 - t1))
            print("=========== Finish Function [" + some_function.__name__ + "] ==========")

    return wrapper


def decor_rundetails(some_function):
    """
    Outputs the details args/time a function takes to execute
    """

    @wraps(some_function)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        try:
            print("========== Running Function [" + some_function.__name__ + "] ==========")
            print("=== [" + some_function.__name__ + "] args: " + str(inspect.getargspec(some_function)))
            print("=== [" + some_function.__name__ + "] stdout:")
            return some_function(*args, **kwargs)
        finally:
            t2 = time.time()
            print("=== [" + some_function.__name__ + "] Time usage: " + time_interval_from_secs(t2 - t1))
            print("=========== Finish Function [" + some_function.__name__ + "] ==========")

    return wrapper


def time_interval_from_secs(runtime_sec):
    """
    get formated string of time length from seconds
    """
    if runtime_sec < 60:
        return str(round(runtime_sec, 3)) + " s"
    elif runtime_sec < 60 * 60:
        minutes = runtime_sec // 60.0
        seconds = round(runtime_sec - minutes * 60, 3)
        return str(minutes) + " min " + str(seconds) + " s"
    elif runtime_sec < 60 * 60 * 24:
        hours = runtime_sec // (60.0 * 60)
        minutes = (runtime_sec - hours * 60 * 60) // 60.0
        seconds = round(runtime_sec - hours * 60 * 60 - minutes * 60, 3)
        return str(hours) + " hrs " + str(minutes) + " min " + str(seconds) + " s"
    else:
        days = runtime_sec // (60.0 * 60 * 24)
        hours = (runtime_sec - days * 60 * 60 * 24) // (60 * 60)
        minutes = (runtime_sec - days * 60 * 60 * 24 - hours * 60 * 60) // 60.0
        seconds = round(runtime_sec - days * 60 * 60 * 24 - hours * 60 * 60 - minutes * 60, 3)
        return str(days) + " days " + str(hours) + " hrs " + str(minutes) + " min " + str(seconds) + " s"


def print_sourcecode(func):
    """
    print function source code
    break into lower level if wrapped with decorator
    """

    def func_extract(func):
        if func.__closure__ is None:
            return func
        else:
            def extract_wrapped(decorated):
                closure = (c.cell_contents for c in decorated.__closure__)
                return next((c for c in closure if isinstance(c, FunctionType)), None)

            return func_extract(extract_wrapped(func))

    for line in inspect.getsourcelines(func_extract(func))[0]:
        print(line.strip("\n"))


def waterfall_allocation(alloclist, allocnum):
    """
    waterfall_allocation(alloclist, allocnum)
    
    waterfall allocation(non negative basis)
    allocnum should <= sum(alloclist) , anything more from allocnum will be dropped
    """
    alloclist = np.array(alloclist)
    if allocnum > sum(alloclist):
        print("Warning: num should <= sum(allocate list)!")
    if any(alloclist < 0):
        # negative basis
        if sum(alloclist) > allocnum:
            res = []
            res.append(min(alloclist[0], allocnum))
            if len(alloclist) > 1:
                for n in alloclist[1:-1]:
                    res.append(min(n, allocnum - sum(res)))
                res.append(allocnum - sum(res))
        else:
            # sum(alloclist) == allocnum
            res = list(alloclist)
        return res
    else:
        res = list(map(lambda x, y: y if x <= 0 else y - x, np.cumsum(alloclist) - allocnum, alloclist))
        res = [max(x, 0) for x in res]
        return res


def list_breakdown(longlist, num_in_each_list):
    """
    breakdown long list to short list
    """
    list_bkdn = []
    if len(longlist) <= num_in_each_list:
        list_bkdn.append(longlist)
    else:
        for i in range(len(longlist) / num_in_each_list):
            list_bkdn.append(longlist[i * num_in_each_list:(i + 1) * num_in_each_list])
        if len(longlist[(i + 1) * num_in_each_list:]) > 0:
            list_bkdn.append(longlist[(i + 1) * num_in_each_list:])
    return list_bkdn


def nan_sum(list_of_numbers):
    """
    sum list/0 if nan
    """
    return sum([x for x in list_of_numbers if x == x])


def fit_distribution(series, dis_type="norm", plot=True, bins=100):
    """
    fit_distribution(series,dis_type="norm",plot=True,bins=100)
    
    fit given series from given distribution
    series na omitted
    return fitted parameters: MLE, location(mean), scale(variance)
    
    support:
        normal
        student t
        gamma
        pareto
        powerlaw
        beta
    """
    import scipy.stats
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(min(series), max(series), bins)

    if dis_type == "norm":
        params = scipy.stats.norm.fit(series)
        pdf_fitted = scipy.stats.norm.pdf(x, *params)
    elif dis_type == "t":
        params = scipy.stats.t.fit(series)
        pdf_fitted = scipy.stats.t.pdf(x, *params)
    elif dis_type == "gamma":
        params = scipy.stats.gamma.fit(series)
        pdf_fitted = scipy.stats.gamma.pdf(x, *params)
    elif dis_type == "pareto":
        params = scipy.stats.pareto.fit(series)
        pdf_fitted = scipy.stats.pareto.pdf(x, *params)
    elif dis_type == "powerlaw":
        params = scipy.stats.powerlaw.fit(series)
        pdf_fitted = scipy.stats.powerlaw.pdf(x, *params)
    elif dis_type == "beta":
        params = scipy.stats.beta.fit(series)
        pdf_fitted = scipy.stats.beta.pdf(x, *params)
    else:
        print("Distribution type not exist!")
        return None

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(x, pdf_fitted, 'b-')
        plt.hist(series, bins=bins, normed=True, alpha=.3)
        plt.title("Distribution type: " + dis_type + " | Parameters: " + str([round(y, 2) for y in params]))
        plt.grid(ls="--", alpha=0.7)
        plt.show()

    return params


def is_number(string):
    """
    test if str can convert to number
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def allocate_int(intnum, intlist, round_type="HALFUP"):
    """
    allocate integer to a list by portion

    Should be all positive int
    allocate by portion/pro rata
    """
    if len(intlist) == 0:
        return []
    factor = float(intnum) / float(sum(intlist))
    if round_type == "HALFUP":
        res = [round(float(x) * factor) for x in intlist]
    elif round_type == "DOWN":
        res = [int(float(x) * factor) for x in intlist]
    else:
        raise Exception("False Roundup Type")

    res[-1] = intnum - sum(res[:-1])
    return res


class Logger(object):
    """
    Logger output
    output to both terminal and log file
    """

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass

    def close(self):
        self.log.close()


def execfile_logger(python_path, log_path):
    """
    execfile wrapper
    """
    default_stdout = sys.stdout
    mylog = Logger(log_path)
    sys.stdout = mylog
    # Run files
    try:
        exec(compile(open(python_path, "rb").read(), python_path, 'exec'), globals())
    except:
        print("------------- Error Message ------------- ")
        print(sys.exc_info())

    sys.stdout = default_stdout
    mylog.close()


def folder_synchronizer(root_src_dir, root_dst_dir, log=[], file_type=[".py"], ignore_folders=[".git"]):
    """
    from root_src_dir to root_dst_dir
    log is a list with log strings
    file_type use .XX
    will ignore folders and sub folder/files 
    """
    for src_dir, dirs, files in os.walk(root_src_dir):
        for ignore in ignore_folders:
            if ignore in src_dir:
                # ignore .git folder files
                pass
            else:
                dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
                # make folder
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                #     
                for file_ in files:
                    # copy if python file
                    if "." + file_.split(".")[-1] in file_type:
                        src_file = os.path.join(src_dir, file_)
                        dst_file = os.path.join(dst_dir, file_)

                        if os.path.exists(dst_file):
                            # judge modfification time 
                            src_file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(src_file))
                            dst_file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(dst_file))

                            if src_file_mtime > dst_file_mtime:
                                #
                                log.append(
                                    "Update [" + dst_file + "] --- Version [" + str(dst_file_mtime) + " ===>> " + str(
                                        src_file_mtime) + "]")
                                os.remove(dst_file)
                                shutil.copy(src_file, dst_dir)
                            else:
                                # modified later or same / no change
                                pass
                        else:
                            # create if not exist
                            src_file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(src_file))
                            log.append("Create [" + dst_file + "] --- Version [" + str(src_file_mtime) + "]")
                            shutil.copy(src_file, dst_dir)
    log.append("Folder up to date! [" + root_dst_dir + "]")
    return log


def string_breakdown(line, sep_numlist, sep_type="index"):
    """
    seperate str line by index list
    add 0 in front of the list if using interval!
    """
    if sep_type == "index":
        return [line[sep_numlist[i]:sep_numlist[i + 1]] for i in range(len(sep_numlist) - 1)] + [
            line[sep_numlist[-1]:]]
    elif sep_type == "interval":
        temp_list = np.cumsum(sep_numlist)
        return string_breakdown(line, temp_list, listtype="index")


def get_nextminute(minutes=1):
    """
    get next minute time string
    """
    return (datetime.datetime.now() + datetime.timedelta(minutes=minutes)).strftime("%H:%M")


def get_today():
    return datetime.datetime.now().strftime("%Y%m%d")


def string_to_unicode(string):
    """transfer str to unicode in py2"""
    return string


