from PyQt5.Qt import qApp

def print_to_log(*args):
    string = " ".join([str(i) for i in args])
    try:
        qApp.main_window.log_te.append("\n" + string)
    except AttributeError:
        print(string)


def print_to_tb(tb = None, *args):
    string = " ".join([str(i) for i in args])
    if tb is None:
        print_to_log(args)
    else:
        tb.append("\n" + string)

def print_log_header(class_name= "next function widget"):
    total_length = 80
    star_length = (total_length - len(class_name)) // 2
    string = "=" * star_length + class_name + "=" * star_length
    print_to_log(string)

def print_tb_header(text_browser=None,run_num = 1):
    header_text = "the "+str(run_num)+" run start"
    total_length = 55
    line_length = (55-len(header_text))//2
    print_to_tb(text_browser, "=" * line_length, header_text, "=" * line_length)