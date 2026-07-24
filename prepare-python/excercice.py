#Ex1: Viết chương trình khai báo thông tin của một sinh viên gồm tên, tuổi và chuyên ngành.
# Sau đó hiển thị thông tin theo định dạng đẹp bằng f-string và in tên ở dạng chữ hoa
name = "Alice"
age = 20
major = "Computer Science"

print(f"Name : {name}")
print(f"Age  : {age}")
print(f"Major: {major}")

print(f"\nHello! My name is {name} and I am {age} years old.")
print(f"Uppercase name: {name.upper()}")

#Ex2:
text = "   Python Programming Language   "
text = text.strip()
text = text.lower()
text = text.replace("programming", "coding")

print("Processed string:")
print(text)

print("\nFirst 6 characters:")
print(text[:6])

print("\nSplit result:")
print(text.split())

#Ex3:

full_name = "   nguyen van a   "
birth_year = "2004"
salary = 1500.75

full_name = full_name.strip().upper()
name_parts = full_name.split()
last_name = name_parts[0]
first_name = name_parts[-1]
birth_year = int(birth_year)
age = 2026 - birth_year
bonus = salary * 0.2
total_income = salary + bonus

print(
    f"Full name      : {full_name}\n"
    f"Last name      : {last_name}\n"
    f"First name     : {first_name}\n"
    f"Age            : {age}\n"
    f"Salary         : {salary}\n"
    f"Bonus          : {bonus}\n"
    f"Total income   : {total_income}"
)
print()

print(f"Age from 18 to 30: {18 <= age <= 30}")
print(f"Salary is float : {isinstance(salary, float)}")