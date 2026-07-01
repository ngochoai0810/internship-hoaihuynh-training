

# Day 30/6
- Add List Items:
      - Insert():
      - Append Items: to add item to the end of the list
      - Insert Items: to insert a list item at a specified index
      - Extend List: 
         + to append fr another list to this list 
         + can add any object (tuple, sets) Ex: add  a tuple to a list

- Remove Specified Item
      + remove(): remove fl name
      + pop(): remove fl index  
         * if not specify index, the pop remove the last item
      + del() : same as pop but different syntax : del thislist[0]
         * can delete completely
      + clear(): empties the list, the list still remain but it has no content
      + 
   - Loop List
      - for loop: 
         + range(), len()
      - while loop:
   - The syntax
      newlist = [expression for item in iterable if condition == True]
   
- Sort Lists
   - sort(): will sort the list alphanumerically
   - sort(reverse = True) : list descending
   - sort(key = myfunc)
   - reverse()
- Copy Lists:
   - copy(): 
   - list() : the same
   - slice(:): empty start index and end index to have a same feature
- Join Lists:
   - use + operator
   - extend() : list1.extend(list2)
   -

# Day 1/7

## Basic theory

Dictionary items are ordered, changeable, and do not allow duplicates
Dictionary Length
Can be any data type in dict items

1. Access Items
    - thisdict[key]
    - get()
    - keys() : get all the keys
    - values(): get all the values
    - items(): return each item in a dict , as tuple in a list
2. Change Values
    - update(): 
3. Adding Items
    - update() : 
4. Remove Items:
    - pop()
    - popitem() : remove last items
    - del():  
        - del thisdict[key]: remove item wth specified key name
        - del thisdict: del completely
    - clear(): emty dict
5. Loop Through a Dict:
    - for x in dict: print key name
    - for x in dict.value(): print values
    - for x, y in thisdict.items():
        print(x, y)
    -counter
6. Copy Dictionary
    - copy()
    - dict()
    -shallow & deepcopy
7. Nested Dictionaries
    - a dict can contain dicts
    - items()
8. Default Dict
    -
9. Type Hint:
def function(parameter: kiểu_dữ_liệu) -> kiểu_trả_về:

10. Any: 
    mypy can't detect error
    reason use Any too much can mypy cant see error, cant catch error
    
11. dict[str, Any] & TypeDict
    = dict[key: str, value: Any]
    dict[str, Any] = JSON dont understand 
    TypeDict = fixed form

    - TypeDict
        class User(TypeDict):
            name:str
            age : int
        user : User = {
            "name": "Hoai",
            "age": 20
        }
Dict comprehension
12. mypy 
Dict comprehension