#install.packages("png")
library ("png")

#Записать одну цифру в PNG файл
#digidData - матрица, содержащая одну строку данных в формате MNIST (784 столбца)
#file - имя файла
mnist.writeSingleDigit=function (digitData,file) {
  writePNG(t(matrix((1.0-digitData/256),ncol=28)),file)
}

#Записать одну цифру в PNG файл
#data - матрица данных MNIST
#row - номер строки (цифры) в матрице данных
#file - имя файла
mnist.writeSingleDigitByRowNum=function (data, row ,file) {
  mnist.writeSingleDigit (unlist(data[row,]),file)
}

#Записать N цифр в PNG файл, начиная со startRow. в виде одной строки
#data - матрица данных MNIST
#startRow - номер строки (цифры) в матрице данных
#N - количество строк (цифр) для записи в файл
#file - имя файла
mnist.writeDigitsToOneRowFile=function (data, startRow, N ,file) {
  m<-matrix(nrow = 28) #dummy matrix column
  for (i in startRow:(startRow+N)){
    row<-unlist(data[i,])
    m1<-t(matrix(1.0-row/256,nrow=28))
    m<-cbind(m,m1)
  }
  m<-m[,-1]
  writePNG(m,file)
}


#Сформировать матрицу данных для PNG файла  N цифр, начиная со startRow. в виде одной строки
#data - матрица данных MNIST
#startRow - номер строки (цифры) в матрице данных
#N - количество строк (цифр) для записи в файл

mnist.getDigitsToOneRowMatrix=function (data, startRow, N) {
  m<-matrix(nrow = 28) #dummy matrix column
  for (i in startRow:(startRow+N)){
    row<-unlist(data[i,])
    m1<-t(matrix(1.0-row/256,nrow=28))
    m<-cbind(m,m1)
  }
  m<-m[,-1]
  return(m)
}

#Записать последовательный набор цифр в PNG файл, начиная со startRow. в виде нескольких строк
#data - матрица данных MNIST
#startRow - номер строки (цифры) в матрице данных
#outputDigitsInColumn - количество строк (цифр) для записи в одну строку файла
#outputRows - количество строк в выходном файле. 
#Общее кол-во выгружаемых данных = outputDigitsInColumn*outputRows
#file - имя файла
mnist.writeDigitsToMultipleRowFile=function (data, startRow, outputDigitsInColumn,
                                              outputRows ,file) {
  outRows=mnist.getDigitsToOneRowMatrix(data, startRow,outputDigitsInColumn)
  #получили первую строку цифр
  
  #формируем остальные outputRows-1 строк
  for (j in 1:(outputRows-1)){ 
    outRowTmp=mnist.getDigitsToOneRowMatrix(data, (j*outputDigitsInColumn+1),outputDigitsInColumn)
    outRows<-rbind(outRows,outRowTmp)
  }
  writePNG(outRows,file)
}