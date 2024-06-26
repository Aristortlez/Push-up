#Type Conversion การแปลงชนิดข้อมูล
x = 10
y = 3.14
z = "20"

# บวกเลข
result = str(x)+z
z=float(z)
z=z+50 
# "20" => 20
# 10 => "10"
# 10+3.14 = 13.14
# "10" + "20" = "1020" => result
print(type(x))
print(type(y))
print(type(z))
print(float(z)) # string => float float(z)
print(str(y)) # float => string str(y)
print(float(x)) # int => float
print(int(y)) # float => int

# string => int x+int(z)
# int => string str(x)+z
# string => float float(z)
# float => string str(y)
# int => float
# float => int
