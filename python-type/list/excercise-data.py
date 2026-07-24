raw_folder = ["api ", " dAta", "Ml "]
clean_folder = [x.strip().lower() for x in raw_folder]
print(clean_folder) 

newlist = [x for x in range (10) if x < 5]
print(newlist)

newlist = [x.upper() for x in raw_folder]

thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(thislist)