thisdict = {
    "brand" : "Ford",
    "electric" : False,
    "model" : "Mustang",
    "year" : 1964,
    "colors" : ["red", "white", "blue"]
}
print(thisdict)
print(thisdict["brand"])
print(len(thisdict))
print(type(thisdict))

thisdict = dict(name = "john", age = 36, country = "Norway")
print(thisdict)

x = thisdict.get("age")
print(x)

x = thisdict.keys()
print(x)

y = thisdict.values()
print(y)
thisdict["age"] = 40
print(y)
z = thisdict.items()
print(z)

if "age" in thisdict :
    print("Yes")

thisdict.update({"age" : 100})
thisdict.update({"color" : "red"})
print(thisdict)

# thisdict.pop("color")
del thisdict["color"]
print(thisdict)

for x in thisdict.keys(): 
    print(x)

mydict = dict(thisdict)

child1 = {
    "name" : "Emil",
    "year" : 2004

}
child2 = {
    "name" : "Tobias",
    "year" : 2007
}

myfamily = {
    "child1" : child1,
    "child2" : child2
}
print(myfamily)

print(myfamily["child2"]["name"])

for x, obj in myfamily.items():
    print(x)
    for y in obj:
        print(y + ':', obj[y])