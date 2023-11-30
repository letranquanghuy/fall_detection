# class Dog():
#     breed = "Mina"
#     def __init__(self, breed):
#         # self.breed = breed
#         pass


# my_dog = Dog("Hello")
# print(type(my_dog))
# my_dog.breed = "hello world"
# print(my_dog.breed)

# # Dog.breed = "Lucky"
# print(Dog.breed)


class Animal():
    def eat(self):
        print("I am eating")

class Dog(Animal):
    def __init__(self):
        super().__init__("Mina")


my_dog = Animal()
my_dog.eat()

a=[1, 2, 3]
a.pop(-1)
print(a)