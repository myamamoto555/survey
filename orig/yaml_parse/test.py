# coding: utf-8
char = u'\u2580'

print char
fout = open("test.txt", "w")
fout.write(char.encode("utf-8"))

fout.close()
