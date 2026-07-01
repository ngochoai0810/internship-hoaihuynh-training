import copy
from collections import defaultdict
from collections import Counter
user ={
    "name": "John",
    "age": 30,
    "config": {
        "theme": "dark", "version": 1.0}
}

shallow_user = user.copy()
shallow_user["config"]["theme"] = "light"
print(user["config"]["theme"])  

user["config"]["theme"] = "dark"

deep_user = copy.deepcopy(user)
deep_user["config"]["theme"] = "light"
print(user["config"]["theme"])

prices_in_usd = {'room1': 100, 'room2': 150, 'room3': 200}
prices_in_vnd = {k: v * 25000 for k, v in prices_in_usd.items()}
print(prices_in_vnd)

groups = defaultdict(list)
groups['students']

print(groups)

text = "hello world hello everyone"
words = text.split()
word_count = Counter(words)
print(word_count)

print(word_count.most_common(1))

