import re

file_path = r"E:\bachelor_artical_related\hc\201911.txt"
file = open(file_path,"r")
result_file = open(file_path.replace("txt","tsv"),"w")
result_dic_file = open(file_path.replace(".txt","_dic.tsv"),"w")
file_text = file.read()
file.close()

regex = re.compile(r"\nPT.*?\nER", re.DOTALL)
regex_de = re.compile(r"\nDE(.*?)\n")
redex_ti = re.compile(r"\nTI(.*?)\n")

result = regex.findall(file_text)
print(len(result))
word_dict = {}
result_lines = ["title\tkeywords\n"]

for i, tup in enumerate(result):
    de_result = regex_de.findall(tup)
    if de_result != []:
        ti_result = redex_ti.findall(tup)
        result_lines.append(ti_result[0]+"\t"+"\t".join(de_result[0].split("; "))+"\n")
        key_word = de_result[0].split("; ")
        for j in key_word:
            j = j.strip()
            try:
                word_dict[j] += 1
            except KeyError as e:
                word_dict[j] = 1

print(word_dict)
for i in word_dict.items():
    item = i[0] + "\t" + str(i[1]) +"\n"
    result_dic_file.write(item)


result_file.writelines(result_lines)
result_file.close()
result_dic_file.close()