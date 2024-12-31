names = []
contact_numbers = []
num = int(input("Enter the total number of contacts you want to save: "))
for i in range(num):
    name = input("Name: ")
    contact_number = int(input("Contact Number: ")) 
    names.append(name)
    contact_numbers.append(contact_number)
print("\nName\t\t\tContact Number\n")
for i in range(num):
    print("{}\t\t\t{}".format(names[i], contact_numbers[i]))
search_term = input("\nEnter search term: ")
print("Search result:")
if search_term in names:
    index = names.index(search_term)
    contact_number = contact_numbers[index]
    print("Name: {}, Phone Number: {}".format(search_term, contact_number))
else:
    print("No records")
delete_term = input("\nEnter delete term: ")
print("delete result:")
if delete_term in names:
    index = names.index(delete_term)
    contact_number = contact_numbers[index]
    print("Name: {}, Phone Number: {}".format(delete_term, contact_number))
else:
    print("No records")
