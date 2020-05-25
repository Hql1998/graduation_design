import os


def countFileLines(filename):
    count = 0;
    handle = open(filename, 'rb')
    for line in handle:
        count += 1;
    return count;


def listdir(dir, lines):
    files = os.listdir(dir)  # 列出目录下的所有文件和目录
    for file in files:
        filepath = os.path.join(dir, file)
        # if os.path.isdir(filepath):  # 如果filepath是目录，递归遍历子目录
        #     listdir(filepath, lines)
        # elif os.path:  # 如果filepath是文件，直接统计行数
        if os.path:  # 如果filepath是文件，直接统计行数
            if os.path.splitext(file)[1] == '.py':
                lines.append(countFileLines(filepath))
                print(file + ':' + str(countFileLines(filepath)))


lines = []
dir = 'E:\python\graduation_design'
listdir(dir, lines)
dir = 'E:\python\graduation_design\GUI_part'
listdir(dir, lines)
print('total lines=' + str(sum(lines)))
