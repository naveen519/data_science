a=10
a
b=20
b
a+b

id = c(1,2,3,4,5)
age = c(22,31,34,40,45)
gen = c("M","F","M","F","M")

df1 = data.frame(id,age,gen)
id = c(1,2,3,6,7)
sal = c(100,200,300,400,500)
desg = c("A","B","A","B","A")

df2 = data.frame(id,sal,desg)

df1
df2

### inner merge 

df3 = merge(x=df1,y=df2, by ="id") ## inner merge
df3
df3 = merge(x=df1,y=df2, by ="id", all.x=T) ## left merge
df3

df3 = merge(x=df1, y=df2, by="id", all.y = T) ## right merge
df3
df3 = merge(x=df1, y=df2, by="id", all.x=T, all.y = T)
df3

age = 30 
if(age <= 25){ 
  agegrp = "0-25"
}else if(age>25 & age<=35){ 
   agegrp = "26-35" 
  }else agegrp = "35+"

agegrp

age = c(21, 36, 49, 54)

for (i in 1:length(age)){
  if(age[i] <= 25){ 
    agegrp[i] = '0-25'
  }else if( age[i] >25 & age[i] <=30){ 
    agegrp = '26-30'  
  }else if( age[i] >30 & age[i]<=40){ 
    agegrp[i] = '31-40'
  } else agegrp[i] ='40+'
}

agegrp
age = c(20,30,40)
agegrp =c()
for (i in 1:length(age)){
  if(age[i] <=25){ 
    agegrp[i] = '0-25'
  }else if( age[i] >25 & age[i] <= 30){ 
    agegrp[i] = '26-30'  
  }else if( age[i] > 30 & age[i]<=40){ 
    agegrp[i] = '31-40'
  } else agegrp[i] ='40+'
}

agegrp

summary(df3)

summary((df3$age))

table(df3$gen)
table(df3$gen, df3$desg)


table(df3$gen, useNA = "always")

seq(2,10,2)

 a = 1:10
 ?seq

 a = seq(2,100, length.out = 10)
 
 ## 
a = c("M","F","M","F")
unique(a)
### sort 

a = c(10,89,66,32,14,19)
sort(a) ## ascending order 
sort(a, decreasing = T)

## sorting data frame 

df3

df3 = df3[sort(df3$id, decreasing = T) ,]
df3

df3 = df3[,sort(colnames(df3)) ]
df3
a = c("apple","microsoft","google","facebook")
b = sort(a)
b

df3 = df3[,c(4,2,1,3,5)]
df3
d = c(20,86,44, 69, 32,NA,NA)
sort(d) 
df3

### 
df2 = na.omit(df3)
df2

### 

is.na(d)

d[is.na(d)] = 0
d

## dataframe functions 
dim(df3) ## number of rows and num of columns 
nrow(df3) ## number of rows 
ncol(df3) ## number of columns 
colnames(df3) ## list of column names 
dfc_names = colnames(df3)
names(df3) = c("id","designation","age","gender","salary")
### numeric functions 

round(10.73, 1)
round(10.993,2)

## ceiling and floor 

ceiling(10.99)
ceiling(10.1)
floor(9.99)
floor(9.01)
trunc(10.66)
trunc(10.99)

## character functions
a = "apple"
 toupper(a)
tolower(a) 
length(a)
nchar(a)
b = "google"
nchar(b)

b = c("apple","microsoft","google","facebook")
nchar(b)

substr(a, 2,4)

date = Sys.Date()

date

year = substr(date,1,4)

year

a = "Firstname Lastname"
x = unlist(strsplit(a," "))
x[1]
date
d = as.character(date)

y = strsplit(date,"-")

d

### 



d = sub(d,pattern = "-",replacement = "/")
d
d = gsub(d, pattern = "-", replacement = "/")
d

strsplit(d,"/")


date = Sys.Date()

d = as.character(date)

strsplit(d, "-")

##  

## Data manipulations

 a= 4 
 sqrt(a)
 log(a)
 log2(a)
 log10(a) 
 
 ## Stat functions 
 a = c(20,30,40,50,60)
 mean(a)
 a = c(20,30,40,50,60,NA,NA)
mean(a,na.rm = T) 

median(a, na.rm = T)
min(a)
max(a) 
p = range(a)
p[1]
var(a)
sd(a)
### 
exp(a)

b = log(10)
b
a = exp(b)
a

## 

k = c(rep("a",10), rep("b",4))
k

## matrices 

a = c(10,20,30,40)
b = matrix(a,nrow=2, ncol=2)

d = matrix(seq(2,20,length.out = 10),nrow = 5, ncol=2)
d
d = matrix(seq(2,20,length.out = 10), nrow=4,ncol=4)
d

d[1,]
d[2,1:2]
d[,2]
sum(d[,1])
sum(d[,2])

m = matrix(seq(2,20,2), nrow=5, ncol=2, byrow = T)
m
a=0
nrow(m)
for( i in 1:nrow(m)){
  a = a + m[i,1]
}
a
sum(m[,1])

colSums(m)
rowSums(m)
m

a=c(2,3,4,5)
b = c(1,2,3,4)

a = matrix(a,nrow = 2, ncol=2)
b = matrix(b, nrow=2,ncol=2)

## date functions - package "lubridate" 
library("lubridate")

date = Sys.Date()

a = "10/10/2017"
class(a)
a_date = as.Date(a,"%d/%m/%Y" )
a_date

a = "10-jun-2017"

a_date = as.Date(a,"%d-%b-%Y")
a_date

a = "10-08-18"

a_date = as.Date(a,"%d-%m-%y")
a_date

a = "10-08-89"
a_date =as.Date(a,"%d-%m-%y", origin = "2000-01-01" )
a_date

month(a_date)
year(a_date)
day(a_date)
week(a_date)
weekdays(a_date)
?weekdays

a = as.Date("2018-08-26")
b = as.Date("2018-08-01")

a-b

a = as.Date("2018-01-26")
b = as.Date("2018-08-01")

difftime(b,a, units = "weeks")

### install packages 

library("RODBC")
install.packages("RODBC")

mydbcon = odbcConnect(dsn = "mydb", uid ="a123", pwd="xyz123")

mydbcon = odbcConnect(dsn="mydb",uid = "poweruser")

dfa = sqlQuery(channel = mydbcon ,query = "select * from mydb.table " )

library(sqldf)
sqldf()
sqldf("select * from df3")

sqldf( "select a.id, b.age, b.sal, a.gen from df1 a , df2 as b 
       join  on a.id = b.id ")


library(dplyr)

## Select 

select(df3, id, age, gen)

## filter() 

filter(df3, gen == "F")
filter(df3, gen=="F" & sal >=100)

## mutate 

setwd("D:/AP/Dtrees")
churn = read.csv("Churn.csv")
head(churn, 10)

## select 

churn2 = select(churn, Day.Mins , Eve.Mins, Night.Mins)
churn3 = filter(churn, Day.Mins >= 200)
churn3 = filter(churn, Day.Mins >=200 & State %in% c("NY","LA","CA"))

## mutate 

churn4 = mutate(churn, usage = Day.Mins+Eve.Mins+Night.Mins)
churn5 = transmute(churn, usage = Day.Mins + Eve.Mins+Night.Mins)

summarise(churn)

churn6 = arrange(churn, Day.Mins)

churn6 = arrange(churn, Day.Mins, desc(Eve.Mins))


id = c(1,2,3,4)
age =c(10,20,30,40)
gen=c("M","F","M","F")

df1 = data.frame(id,age,gen)

id = c(2,3,4,5)
dept = c("A","B","A","B")
sal = c(100,200,300,400)
df2 = data.frame(id,dept,sal)
inner_join(df1,df2,"id")
left_join(df1,df2,"id")
right_join(df1,df2,"id")
full_join(df1,df2,"id")

id = c(1,2,3,4,1)
sal = c(100,200,300,400,100)
age = c(20,30,40,50,20)
df4 = data.frame(id,sal,age)
df5 = distinct(df4)

id = c(1,2,3,4,1)
sal = c(100,200,300,400,500)
age = c(20,30,40,50,20)

dfx = data.frame(id,sal, age )

distinct(dfx, id,  .keep_all = T)

### mutate 

chrunx = mutate(churn, usage = Day.Mins+Eve.Mins+Night.Mins,
                  charges = usage *1.08)


## summarise 

?summarise

summarise(churn, mean(Day.Mins) )

### Sample 

churn4 = sample_n(churn, 100)
churn8 = sample_frac(churn, 0.1)

region = c(rep("A",3), rep("B",3))
month = c("Jan","Feb","Mar", "Jan","Feb","Mar")
sales = c(100,200,300,400,200,100)

dfk = data.frame(region, month, sales)
dfk
library(reshape)

dfy = reshape(dfk, timevar = "month", idvar="region", direction = "wide")

varnames = colnames(dfy)
varnames

for(i in 1:length(varnames)){ 
  varnames[i] = gsub(varnames[i],pattern = "sales.",replacement = "")
}
varnames
names(dfy) = varnames
colnames(dfy)
