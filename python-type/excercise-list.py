numbers = [10, 20, 30, 40, 50]

print("Third element :", numbers[2])
print("Length of list :", len(numbers))

isEmpty = len(numbers) == 0
print("Is list empty :", isEmpty)

list=[100,50,400,500]

list[1] = 200
print(list)

list.append(600)
print(list)
list.insert(2,300)
print(list)
list.remove(600)
print(list)
list.pop(0)
print(list)

numbers = [10, 20, 30, 40, 50]
print("Sum:", sum(numbers))
print("Average:", sum(numbers)/len(numbers))


factors = [2, 3, 5, 7]
product = 1

for x in factors:
    product *= x

print(f"Product: {product}")

