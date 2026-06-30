thislist = ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
print(thislist)
print(thislist[-1])
print(thislist[2:5])
if "apple" in thislist:
    print("Apple")
thislist[1] = "blackcurrant"
print(thislist)
thatlist =["apple", "banana", "cherry"]
# thatlist.insert(2, "watermelon")
# thatlist.append("watermelon")
thatlist.insert(1, "orange")
print(thatlist)
for x in thatlist:
    print(x)
for i in range(len(thatlist)):
    print(thatlist[i])

i = 0
while i < len(thatlist):
    print(thatlist[i])
    i+= 1

newlist =[x for x in thatlist if "a" in x]
print(newlist)

