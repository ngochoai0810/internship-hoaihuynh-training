x = "awesome"
def myfunc():
  global x
  x = "fantastic"  
myfunc()
print ("Python is " + x)

for x in "banana":
  print(x)
  
a = "Hello, World!"
print(len(a))

print("Hello" in a)
if "Hello" in a:
  print("Yes, 'Hello' is present.")

b = "Hello, World!"
print(b[2:5])
print(b[:5])
print(b[2:])
print(b[-5:-2])
print(b.upper())
print(b.strip())
print(a.replace("H", "J"))
print(a.split(","))
price = 9
txt =f"The price is {price:.2f} dollars"
print(txt)
txt = "The price \"vik\"from a variable"
print(txt)

class myclass():
  def __len__(self):
    return 0
myobj = myclass()
print(bool(myobj))

def my_function():
  return True
print(my_function())

x = 200
print(isinstance(x, int))
numbers = [1, 2, 3, 4, 5]
if(count:=len(numbers)) > 3:
  print(f"List is too long {count} elements")
  
X = 5
print(1<X<10)
print(1<X and X<10)