import re
import matplotlib.pyplot as plt

with open("mlp.txt","r") as f:
    with open("mlp.csv","w") as w:
        lines = f.readlines()
        for line in lines:
            s = re.findall(r"\d+\.?\d*", line)
            s = [float(i) if '.' in i else int(i) for i in s]
            print(s[0],",",s[1],",",s[2],file=w)