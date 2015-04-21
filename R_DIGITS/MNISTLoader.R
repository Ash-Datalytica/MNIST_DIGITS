#There is MNIST loader implementation in the darch package. See http://www.inside-r.org/packages/cran/darch/docs/readMNIST

#Функци считывает данные из MNIST файла в формате CSV с заголовком
#file - полный путь к считываемому файлу
#labels - возвращаема  матрица заголовков, содержаща 1 столбец и число строк, равное числу цифр в исходном файле
#data -  возвращаема матрица данных, содержаща 784 столбца.28*28=784 столбца. их названи pixel0...pixel784
#        число строк равно числу цифр в файле. Строки data соответствуют строкам labels
mnist.loadFromCSV<-function(file){
  
  #Loading train data
  ret<-list() #local function variable
  data <- read.csv(file, header=TRUE) #Считываем данные в двумерную матрицу
  ret$n<-nrow(data)
  ret$l<-data[,1] #названи цифр (labels)
  ret$y <<- RSNNS::decodeClassLabels(mnist.train$l)
  ret$x<-data[,-1] #Точки цифр (оставшиес 28*28=784 столбца. их названи pixel0...pixel784). В виде data.frame
  ret$x<-data.matrix(ret$x, rownames.force=NA) #Convert data.frame to matrix
  #Set colnames for x and y matrices
  colnames(ret$x)<<-paste0('x',1:ncol(ret$x))
  colnames(ret$y)<<-paste0('y',(0:9))
  mnist.train <<- ret #global variable
  
  #Loading test data not implemented
  mnist.test<<- NULL #don't have train data in CSV format
  #   colnames(mnist.test$x)<<-paste0('x',1:ncol(mnist.test$x))
  #   colnames(mnist.test$y)<<-paste0('y',0:9)
  
}

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call: show_digit(train$x[5,]) to see a digit.
# Source: brendan o'connor - gist.github.com/39760 - anyall.org
mnist.loadFromBin <- function(mnistDirectory) {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    #reading the header:
    readBin(f,'integer',n=1,size=4,endian='big') #magic number (2051)
    ret$n = readBin(f,'integer',n=1,size=4,endian='big') #n rows (60000)
    nrow = readBin(f,'integer',n=1,size=4,endian='big') #n pixel rows in one image (28)
    ncol = readBin(f,'integer',n=1,size=4,endian='big') #n pixel columns in one image (28)
    #print (ret$n); print (nrow);print (ncol);
    #reading the data:
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F) #image pixel data
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    ret=list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    l = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    l
  }
  mnist.train <<- load_image_file(paste(mnistDirectory,'train-images.idx3-ubyte', sep=''))
  mnist.test <<- load_image_file(paste(mnistDirectory,'t10k-images.idx3-ubyte', sep=''))
  mnist.train$l <<- load_label_file(paste(mnistDirectory,'train-labels.idx1-ubyte',sep='')) #output decimal digit
  mnist.test$l <<- load_label_file(paste(mnistDirectory,'t10k-labels.idx1-ubyte', sep = ''))  
  mnist.train$y <<- RSNNS::decodeClassLabels(mnist.train$l) #analogue to nnet::class.ind
  mnist.test$y <<- RSNNS::decodeClassLabels(mnist.test$l) #output binary vector
  
  #Set colnames for all used matrices
  colnames(mnist.train$x)<<-paste0('x',1:ncol(mnist.train$x))
  colnames(mnist.train$y)<<-paste0('y',(0:9))
  colnames(mnist.test$x)<<-paste0('x',1:ncol(mnist.test$x))
  colnames(mnist.test$y)<<-paste0('y',0:9)
}

#Example: mnist.show_digit (train$x[20000,])
mnist.show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

#make sample set reducing train and test data
#trainRatio - дол тренировочных даннх, которые надо остаить
#testRatio - ... тестовых ...
mnist.makeSample <- function (trainRatio=0.25, testRatio=1.0) {
  if (trainRatio<1) {
    data <-cbind(mnist.train$l, mnist.train$y, mnist.train$x)
    data <- data[sample(1:mnist.train$n,mnist.train$n*trainRatio),1:ncol(data)] 
    mnist.train$n <<- nrow(data)
    mnist.train$l <<- data[,1]
    mnist.train$y <<- data[,2:(ncol(mnist.train$y)+1)]
    mnist.train$x <<- data[,(ncol(mnist.train$y)+2):(ncol(mnist.train$y)+ncol(mnist.train$x)+1)]
  }
  if (testRatio<1) {
    data <-cbind(mnist.test$l, mnist.test$y, mnist.test$x)
    data <- data[sample(1:mnist.test$n,mnist.test$n*testRatio),1:ncol(data)] 
    mnist.test$n <<- nrow(data)
    mnist.test$l <<- data[,1]
    mnist.test$y <<- data[,2:(ncol(mnist.test$y)+1)]
    mnist.test$x <<- data[,(ncol(mnist.test$y)+2):(ncol(mnist.test$y)+ncol(mnist.test$x)+1)]
  }
}


#Normalaize data to 0_1
#center data if doCenter is TRUE
mnist.normalizeAndCenter <- function(type="0_1", doCenter=TRUE) {
  #is.na(c(NA,'',NULL))
  if (is.null(mnist.train))  {
    print('Error in normalize. No training data')
    return ()
  } 
  if (type == "-1_1") {
    #do norm for train
    normFactor <- (max(mnist.train$x)+1)/2  
    mnist.train$x <<- mnist.train$x/normFactor
    if (doCenter) {
      centerFactor <- 1-1/(2*normFactor) #mean(mnist.train$x)
      mnist.train$x <<- mnist.train$x - centerFactor
    }
    if (is.null(mnist.test))  {
      print('Warning in normalize. No test data')
    } else {
      #do norm for test
      mnist.test$x <<- mnist.test$x / normFactor #Используем тот же normFactor, как и дл train
      if (doCenter) mnist.test$x <<- mnist.test$x - centerFactor #Используем тот же centerFactor, как и дл train
    }
  } else if (type == "-0.5_0.5") {
    #do norm for train
    normFactor <- max(mnist.train$x)+1 #or without "+1"
    mnist.train$x <<- mnist.train$x/normFactor
    if (doCenter) {
      centerFactor <- 0.5-1/(2*normFactor) #mean(mnist.train$x)
      mnist.train$x <<- mnist.train$x - centerFactor
    }
    if (is.null(mnist.test))  {
      print('Warning in normalize. No test data')
    } else {
      #do norm for test
      mnist.test$x <<- mnist.test$x / normFactor #Используем тот же normFactor, как и дл train
      if (doCenter) mnist.test$x <<- mnist.test$x - centerFactor #Используем тот же centerFactor, как и дл train
    }
    
  } else if (type == "0_1") {
    #do norm for train
    normFactor <- max(mnist.train$x)+1 #or without "+1"
    mnist.train$x <<- mnist.train$x/normFactor
    if (doCenter) {
      centerFactor <- -1/(2*normFactor) #mean(mnist.train$x)
      mnist.train$x <<- mnist.train$x - centerFactor
    }
    if (is.null(mnist.test))  {
      print('Warning in normalize. No test data')
    } else {
      #do norm for test
      mnist.test$x <<- mnist.test$x / normFactor #Используем тот же normFactor, как и дл train
      if (doCenter) mnist.test$x <<- mnist.test$x - centerFactor #Используем тот же centerFactor, как и дл train
    }
    
  }
  cat("Used normFactor=", normFactor,",centerFactor=", centerFactor)
}

#Функци подсчета точности модели на основе confusion матрицы
#Удачные предсказани = сумма диагональных элементов матрицы
#Все предсказани = сумма всех элементов матрицы
mnist.modelAccuracy = function (confMatrix) {
  return (sum(diag(confMatrix))/sum(confMatrix))
}
