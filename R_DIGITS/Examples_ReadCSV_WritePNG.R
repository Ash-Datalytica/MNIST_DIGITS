#примеры загрузки данных MNIST из CSV файла и выгрузки их в PNG
#Взяты из статьи http://beyondvalence.blogspot.ru/2014/01/r-classifying-handwritten-digits-mnist.html

mnist.train <- read.csv("../MNIST_DATA/train.csv", header=TRUE) #Считываем данные в двумерную матрицу?
#mnist.train [1:10, ] #Отображаем первые 10 строчек (заголовок + 9 цифр)

#Выгрузим данные в виде PNG файлов
#install.packages("png")
library ("png")
top10<-mnist.train[1:10,] #отбираем первые 10 строк данных
top10.lab <- top10[,1] #названия цифр (Labels)
top10.pix <- top10[,-1] #данные (оставшиеся 28*28=784 столбца. их названия pixel0...pixel784)

#делаем матрицу для изображения цифры №1 и выводим ее в PNG файл
row.1<-unlist(top10.pix[1,])
matrix1<-t(matrix((1.0-row.1/256),ncol=28))
writePNG(matrix1,"../MNIST_DATA/PNG/row1.png")

#выводим изображения цифр №1-10 в две строки. Первая строка содержит цифры№1-5, вторая - №6-10
m3<-matrix(nrow = 28) #dummy matrix column
for (i in 1:5){
  row<-unlist(top10.pix[i,])
  m1<-t(matrix(1.0-row/256,nrow=28))
  m3<-cbind(m3,m1)
}
m3<-m3[,-1]
m4<-matrix(nrow = 28) #dummy matrix column
for (i in 6:10){
  row<-unlist(top10.pix[i,])
  m1<-t(matrix(1.0-row/256,nrow=28))
  m4<-cbind(m4,m1)
}
m4<-m4[,-1]
writePNG(rbind(m3,m4),"../MNIST_DATA/PNG/rows1_10.png")

