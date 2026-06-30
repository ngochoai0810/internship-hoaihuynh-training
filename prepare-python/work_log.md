# Day 26/6

## BASIC THEORY

1. Python is an interpreted language
   - Code is translated and executed line by line at runtime
   - Doesn't produce an executable (.exe)
   - *Reason: statements don't end with ; or , since they execute one by one*

2. Indentation

3. Variables
   - Types: String, integer, float, bool, bytes, complex
   - Rules:
     + Must start with a letter or underscore (_a)
     + Cannot start with a number
     + Only contain 0-9, A-Z, a-z, _
     + Case-sensitive (age != Age != AGE)
     + Cannot be Python keywords
   - Output Variables:
     + (str+str), (int+int), (int,str) but (int+str) is not equal
   - Global var: highest priority level

4. Type Convert
   - Cannot convert complex to other types

5. Slice String
   - str[2:5] -> start at 2 and end at 4

6. Modify Strings
   - .upper() : hel -> HEL
   - .lower() : HELL -> hell
   - .strip() : remove whitespace
   - .replace() : H -> J
   - .split() : substring if it finds ","

7. Concatenation
   - Use {} with f-string

8. Booleans
   - isinstance() : check an object is an integer or not

9. Operators
   - Modulus: %, Exponentiation: x**y
   - Division "/" return float, Floor division "//" return integer
   - Comparison operators
   - Chaining comparison: allow to chain comparison operators
   - Logical operators: and, not, or
   - Assignment: =, +=, -=, *=, /=, //=, %=, **=
   - Ternary: x if condition else y
   - Identity: is, is not -> checks if same object in memory
   - Membership: in, not in -> checks if value exists in sequence
   - Bitwise: &, |, ^, ~, <<, >>


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