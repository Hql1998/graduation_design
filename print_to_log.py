from PyQt5.Qt import qApp

def print_to_log(*args):
    string = " ".join([str(i) for i in args])
    if qApp.main_window is None:
        print(string)
    else:
        qApp.main_window.log_te.append("\n" + string)


def print_to_tb(tb = None, *args):
    string = " ".join([str(i) for i in args])
    if tb is None:
        print_to_log(args)
    else:
        tb.append("\n" + string)
