import subprocess
#import DaiToolkit as tk

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


if __name__ == "__main__":

    pkg_list = ['pympler',
                'empyrical',
                'tushare',
                'PyPDF2',
                'selenium',
                'yfinance',
                'windnd']

    for pkg_name in pkg_list:
        pass
        install_package(pkg_name, install_type='pip')