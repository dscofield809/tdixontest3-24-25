class Student:
    def _init_(self, name, year, school, gpa):
        self.name = name
        self.year = year
        self.school = school
        self.gpa = gpa
    
    def set_name(self, name):
        self.name = name
    
    def set_year (self, year):
        self.year = year

    def set_school(self, school):
        self.school = school

    def set_gpa(self, gpa):
        self.gpa = gpa


Tytrez = Student()

Tytrez._init_("Tytrez Dixon", "Junior", "FMU", 4.0)

print(Tytrez.name)
print(Tytrez.year)
print(Tytrez.school)
print(Tytrez.gpa)
print()

Tytrez.set_name("Robert Dixon")
Tytrez.set_year("Senior")
Tytrez.set_school("UofSC")
Tytrez.set_gpa(3.8)

print(Tytrez.name)
print(Tytrez.year)
print(Tytrez.school)
print(Tytrez.gpa)