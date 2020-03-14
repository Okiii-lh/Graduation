"""
@File       :   test_speak.py
@Contact    :   Okery.github.com

@Modify Time            @Author     @Version    Description
-------------------     -------     --------    ------------
2019/11/11 下午8:41     LiuHe       v1.0        None
"""
import pyttsx3

faces = []

faces1 = [1, 2]

print(faces == faces1)

file_name = "data/name_dat/names.txt"
name_list = []

with open(file_name, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()
        print(lines)
        if not lines:
            break
            pass
            name_tmp = lines
            name_list.append(name_tmp)


for i in name_list:
    print(i)