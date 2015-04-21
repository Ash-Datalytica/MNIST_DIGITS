#For work with MNIST by RSNNS nnet see Example_RSNNS.R
#remove.packages("neuralnet.Debug")
#install.packages("C:/Program Files/R/R-3.1.1/library/neuralnet.Debug")
#install.packages ('neuralnet',type = "source", INSTALL_opts=c("--preclean", "--install-tests","--with-keep.source", "--example"));
#install.packages ('RSNNS')
##install.packages('debug')
#library(debug)
#install.packages("beepr")
#install.packages("Hmisc") #for studying Inputs
library('neuralnet.Debug')# изучаемый пакет
library('RSNNS') #используем функции подготовки данных в loadFromBin()
library(ggplot2) #for nice plots
#library(beepr)
library(Hmisc)
source("MNISTLoader.R") #Подключаем функции чтени данных MNIST


#genegate starweights list for neuralnet with one hidden layer
generateStartWeights <-function (nInput, nHidden, nOutput, type='neauralnet', mean=0, sd=1.22) {
  #Each position in the list is a matrix of weights. First position contains weights fo Input-Hidden layer
  #Second position for Hidden-Output layer.
  #Each matrix has one more row. Why? may be it is for biases?
  #1.22 constant is used due to the recommedations given here^
  #http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
  if (type=='neuralnet') {
    return (list(matrix(data=rnorm((nInput+1)*nHidden, mean=mean, sd = sd), nrow=(nInput+1), ncol=nHidden),
          matrix(data=rnorm((nHidden+1)*nOutput, mean=mean, sd = sd), nrow=(nHidden+1), ncol=nOutput)))
  } else if (type=='LeCun') {
      #According to LeCun's Efficient BackProp, chapter 4.6, formula (16)
      sd2 <- nInput^-0.5
      sd3 <- nHidden^-0.5
      a2 <- 3^0.5*sd2
      a3 <- 3^0.5*sd3
      return (list(matrix(data=runif((nInput+1)*nHidden, min=-a2, max=a2), nrow=(nInput+1), ncol=nHidden),
           matrix(data=runif((nHidden+1)*nOutput, min=-a3, max=a3), nrow=(nHidden+1), ncol=nOutput)))
    } else print ("Unknown type in generateStartWeights.")
}

#draw learning history for neuralnet
#works only if neuralnet.Debug is used (because of $history element)
drawLearningHistory <- function (hist){
  df<- data.frame(hist,row.names = NULL)
  names(df)<-c('step','threshold','error')
  p <-ggplot(df, aes(step) )
  p <- p + geom_line(aes(y=error), colour="red")
  p <- p + geom_line(aes(y=threshold), colour="green")
  p <- p + scale_y_log10()
  #p + theme(legend.position = "bottom", legend.box = "horizontal")
  #p + guides(step = "colorbar", threshold = "legend", error = "legend")
  #p <- p + guides(step = guide_legend("title"), threshold = guide_legend("title"),
  #           error = guide_legend("title"))
  return(p)
}


#start study of neuralnet package. 
#model predicts Lbl ~ x1+...+x784
neuralnet.predictLabelsStudy<- function(){
  start<-Sys.time() 
  #mnist.loadFromCSV("../MNIST_DATA/train.csv") #42000 цифр, 12-13 сек.
  mnist.loadFromBin('../MNIST_DATA/UNZIP/') #60000+10000 цифр, 4-5 сек
  #mnist.train$y [1:10] #Отображаем первые 10 строчек заголовков 
  finish<-Sys.time() ; print("Data loaded"); print (finish-start)
  #ncol(mnist.train$x)#=784
  
  ##проверка загруженных данных. Show Image
  #mnist.show_digit (mnist.train$x[20000,])
  
  ##проверка загруженных данных. выгружаем в PNG формат.
  #source("MNIST2PNG.R") #Подключаем функции записи в PNG файлы
  #mnist.writeSingleDigit(unlist(mnist.train$x[1,]),"../MNIST_DATA/PNG/row1.png") #записать цифру №1 в файл
  #mnist.writeSingleDigitByRowNum(mnist.train$x,2,"../MNIST_DATA/PNG/row2.png") #записать цифру №2 в файл
  #mnist.writeDigitsToOneRowFile (mnist.train$x,1,100,"../MNIST_DATA/PNG/row1_100.png")
  ##записать цифры 1:100 в файл в виде одной строки. 
  #Следуюуща команда выполнетс долго! >20 минут
  #mnist.writeDigitsToMultipleRowFile (mnist.train$x,1,100,420,"../MNIST_DATA/PNG/rows1_42000.png")
  ##записать все цифры (1:(100*420)) в файл в виде матрицы 100*420
  
  
  ##дл отладки оставлем часть данных дл обучени (ускорение)
  mnist.makeSample(trainRatio = 0.1,testRatio=0.1)
  print(paste('Training sample size is',mnist.train$n, "rows."))
  
  mnist.normalize()
  # View(mnist.train$l[1:50])
  # View(mnist.train$x[1:50,])
  # View(mnist.train$y[1:50,])
  # View(cbind(mnist.train$y,mnist.train$x)[1:50,])
  #ncol(mnist.train$x)
  #ncol(mnist.train$y)
  

  
  #Формула в формате neuralnet (кто от кого зависит)
  #myFormula = paste(paste(colnames(mnist.train$y),collapse='+'),'~',
  #                paste(colnames(mnist.train$x),collapse='+'))
  myFormula = paste('Lbl ~',
                    paste(colnames(mnist.train$x),collapse='+'))
  
  tmpData<-cbind(mnist.train$l,mnist.train$x)
  colnames(tmpData)<-c(('Lbl'),paste0('x',1:ncol(mnist.train$x)))
  
  start<-Sys.time()
  #Сеть должна закончить работу, достигнув threshold, иначе она не посчитает матрицу вестов и т.п.
  mnist.neuralnet <- neuralnet (formula=myFormula, 
                                data=tmpData, 
                                hidden = (30), 
                                learningrate=0.1, #a numeric value specifying the learning rate used by 
                                # traditional backpropagation. 
                                #Used ONLY for traditional backpropagation.
                                algorithm='slr', # 'backprop', 'rprop+', 'rprop-', 'sag', or 'slr'.
                                rep=1,
                                threshold = 10, #an integer specifying the threshold for the partial derivatives of the error function  as stopping criteria. Default: 0.01.
                                act.fct = "tanh", #'logistic' (default) or 'tanh'
                                err.fct="ce", #"sse"=’sum of squared errors’ and "ce"=’cross entropy’.
                                linear.output=TRUE,#If act.fct should not be applied to the output neurons set linear output to TRUE (default), otherwise to FALSE.
                                stepmax=250000, #the maximum steps for the training of the neural network. Reaching this maximum leads to a stop of the neural network's training process.
                                lifesign="full", 
                                lifesign.step=100
  )
  finish<-Sys.time(); print('NNet training completed in'); print((finish-start))
  tmpData<-NULL
  
  print(mnist.neuralnet)
  #print(mnist.neuralnetF)
  #plot(mnist.neuralnet)#слишком больша нейросеть
  #mnist.neuralnet$startweights #null
  #View(data.frame(Predicted=mnist.neuralnetF$net.result, Expected=mnist.train$l))
  #mnist.neuralnet$response
  #mnist.neuralnet$covariate
  # res<-mnist.neuralnet$result.matrix #a matrix containing the reached threshold, needed steps, error, AIC and BIC (if computed) and weights for every repetition. Each column represents one repetition
  # View (res)
  # View (res[23000:nrow(res),])
  #mnist.neuralnet$model.list
  #confidence.interval(mnist.neuralnet) # not enough memory
  
    
  
  # Accuracy on training data
  resTrain<-compute(mnist.neuralnet,mnist.train$x)
  #View(resTrain$net.result)
  #Проверка, какой элемент самый минимальный. При обучении по Lbl минимум <0, 
  #считаем, что модель дала ответ 0 вместо отрицательного числа. 
  #По-сути,Немного жульничаем с результатами.
  if (resTrain$net.result[which.min(resTrain$net.result)]<0) {
    #mnist.train$l[which.min(resTrain$net.result)]
    #View(resTrain$net.result[resTrain$net.result<0])
    #View(mnist.train$l[resTrain$net.result[resTrain$net.result<0]])
    resTrain$net.result[resTrain$net.result<0]<-0
    #View(resTrain$net.result)  
  }
  confMatrix1<-table(round(resTrain$net.result),mnist.train$l) # Confusion Matrix
  print("Confusion matrix for training set:")
  print(confMatrix1)
  print(paste("Model accuracy on training set is", round(mnist.modelAccuracy(confMatrix1)*100,4), "percent."))
  print("")
  
  # Accuracy on test data
  resTest<-compute(mnist.neuralnet,mnist.test$x)
  if (resTest$net.result[which.min(resTest$net.result)]<0) {
    #mnist.test$l[which.min(resTest$net.result)]
    resTest$net.result[resTest$net.result<0]<-0
    #View(resTesl$net.result)  
  }
  confMatrix2<-table(round(resTest$net.result),mnist.test$l)
  print("Confusion matrix for test set:")
  print(confMatrix2)
  print(paste("Model accuracy on test set is", round(mnist.modelAccuracy(confMatrix2)*100,4), "percent."))
}

##############################################################################
#study inputs
neuralnet.studyInputs <- function (){
  mnist.loadFromBin('../MNIST_DATA/UNZIP/') #60000+10000 цифр, 4-5 сек
  mnist.makeSample(trainRatio = 1,testRatio=1)
  print(paste('Sdutied sample size is',mnist.train$n, "rows."))
  
  mnist.normalizeAndCenter(type="-0.5_0.5") #normalize train and test data
  
  mean(mnist.train$x)
  sd(mnist.train$x)
  max(mnist.train$x)
  min(mnist.train$x)
  #hist(mnist.train$x, breaks=20) #http://rrus.wordpress.com/tag/%D0%B3%D1%80%D0%B0%D1%84%D0%B8%D0%BA%D0%B0/
  plot(density(mnist.train$x))
  plot(density(mnist.test$x))
  #?density
}
##############################################################################
#start study of neuralnet package. 
#model predicts formula=y0+...y9 ~ x1+...+x784
neuralnet.predictClassStudy<- function(){
  start<-Sys.time() 
  mnist.loadFromBin('../MNIST_DATA/UNZIP/') #60000+10000 цифр, 4-5 сек
  finish<-Sys.time() ; print("Data loaded"); print (finish-start)
  
    ##дл отладки оставлем часть данных дл обучени (ускорение)
  mnist.makeSample(trainRatio = 0.01,testRatio=1)
  print(paste('Training sample size is',mnist.train$n, "rows."))
  
  #mnist.normalizeAndCenter(type="0_1") #normalize train and test data
  
#   #normalization by LeCun for each pixel independently (not the best approach)
#   means <- apply(mnist.train$x,2, mean)
#   #View (means)
#   mnist.train$x <- mnist.train$x - matrix(rep(means,nrow(mnist.train$x)), nrow = nrow(mnist.train$x), ncol=ncol(mnist.train$x), byrow=T)
#   mnist.test$x <- mnist.test$x - matrix(rep(means,nrow(mnist.test$x)), nrow = nrow(mnist.test$x), ncol=ncol(mnist.test$x), byrow=T)
#   sds <- apply(mnist.train$x,2, sd)
#   #View(sds)
#   sds[sds==0] <- 1 #to avoid division by 0
#   #plot(sds)
#   mnist.train$x <- mnist.train$x / matrix(rep(sds,nrow(mnist.train$x)), nrow = nrow(mnist.train$x), ncol=ncol(mnist.train$x), byrow=T)
#   mnist.test$x <- mnist.test$x / matrix(rep(sds,nrow(mnist.test$x)), nrow = nrow(mnist.test$x), ncol=ncol(mnist.test$x), byrow=T)
#   mean (mnist.train$x); sd (mnist.train$x)
#   min (mnist.train$x); max (mnist.train$x)

  #normalization by LeCun for the whole matrix (the best result)
  mean <- mean(mnist.train$x)
  mnist.train$x <- mnist.train$x - mean
  mnist.test$x <- mnist.test$x - mean
  sd <- sd(mnist.train$x)
  mnist.train$x <- mnist.train$x / sd
  mnist.test$x <- mnist.test$x / sd
  mean (mnist.train$x); sd (mnist.train$x)
  min (mnist.train$x); max (mnist.train$x)
  #mnist.show_digit(mnist.train$x[2,])

#  normalization via RSNNS package, gives the result  similar to my mnist.normalizeAndCenter
#    mnist.train$x<-RSNNS::normalizeData(mnist.train$x, type = 'norm') #0_1, center, norm
#    mnist.test$x<-RSNNS::normalizeData(mnist.test$x, type = 'norm')
# #   plot(density(mnist.train$x))
# #   #Set colnames for all used matrices
#    colnames(mnist.train$x)<-paste0('x',1:ncol(mnist.train$x))
#    colnames(mnist.test$x)<-paste0('x',1:ncol(mnist.test$x))
  #View (mnist.train$x)
    
  #Формула в формате neuralnet (кто от кого зависит)
  #formula=y0+...y9 ~ x1+...+x784
  myFormula = paste(paste(colnames(mnist.train$y),collapse='+'),'~',
                  paste(colnames(mnist.train$x),collapse='+'))
  
  nHidden=30
  #We would use logistic activation function, so it's better to prepare start weights of neurons to be near 0.5 for faster learning
   startWeights <- generateStartWeights(ncol(mnist.train$x), nHidden,ncol(mnist.train$y), 
                                      type='neuralnet', mean=0.0, sd=0.1)
  #startWeights <- generateStartWeights(ncol(mnist.train$x), nHidden,ncol(mnist.train$y), type='LeCun')
  #plot(density(startWeights[[1]]));plot(density(startWeights[[2]]))

  start<-Sys.time()
  mnist.neuralnet<-NULL
  #theoretically we should use outputs in (-1,1) to obtain tanh error function advantage
  #logistic error function gives better results, so we use logistic and no hift forthe outputs
  outputShift <- -0.5 #-0.5 shift makes no difference or makes the result a bit worse
  #Сеть должна закончить работу, достигнув threshold, иначе она не посчитает матрицу весов и т.п.
  #View(cbind(2*(mnist.train$y+outputShift),mnist.train$x)) #to make y be in {-1,1}
  mnist.neuralnet <- neuralnet (formula=myFormula, 
                                data=cbind(2*(mnist.train$y+outputShift),mnist.train$x), 
                                hidden = c(nHidden), 
                                algorithm='sag', # 'backprop', 'rprop+'(default), 'rprop-', 'sag', or 'slr'.
                                #learningrate=0.1, #a numeric value specifying the learning rate used by 
                                # traditional backpropagation. Used ONLY for traditional backpropagation.
                                learningrate.limit=list(min=c(1e-8), max=c(0.01)), #default 1e-10, 0.1
                                learningrate.factor=list(minus=c(0.8), plus=c(1.2)), #default values minus/plus= 0.5/1.2
                                rep=1,
                                threshold = 0.2, #an integer specifying the threshold for the partial derivatives of the error function  as stopping criteria. Default: 0.01.
                                act.fct = "logistic", #'logistic' (default) or 'tanh'
                                err.fct="sse", #"sse"=’sum of squared errors’ and "ce"=’cross entropy’.
                                linear.output=TRUE,#If act.fct should not be applied to the output neurons set linear output to TRUE (default), otherwise to FALSE.
                                startweights=startWeights,
                                stepmax=5e05, #the maximum steps for the training of the neural network. Reaching this maximum leads to a stop of the neural network's training process.
                                lifesign="full", 
                                lifesign.step=500
  )
  finish<-Sys.time(); print('NNet training completed in'); print((finish-start));#beep()
    
  print(mnist.neuralnet)
  #summary(mnist.neuralnet)
#   res=mnist.neuralnet$net.result  
#   names(res)="data"
#   View(data.frame(Label=mnist.train$l, Predicted=res, Expected=mnist.train$y))
  #View(mnist.neuralnet$response)
  ##View(res$data)
  ##sum(res$data[1,])
  #mnist.error<-mnist.neuralnet$result.matrix[1]
  #mnist.reached.threshold<-mnist.neuralnet$result.matrix[2]
  mnist.steps<-mnist.neuralnet$result.matrix[3]
  print (paste("Скорость расчета сети:",round(mnist.steps/as.numeric((finish-start), units="mins")),
                "итераций/мин"))  
  h <- mnist.neuralnet$history[[1]]#rep=1
  #View(h)#View how the net was trained
  drawLearningHistory(h)

  # Accuracy on training data
  resTrain<-compute(mnist.neuralnet,mnist.train$x)
  predictedLbl <- (0:9)[apply(resTrain$net.result-outputShift, 1, which.max)] #predicted labels
  #View (cbind(predictedLbl,mnist.train$l))
  confMatrix1<-table(Expected=mnist.train$l,Predicted=predictedLbl)
  print("Confusion matrix for training set:")
  print(confMatrix1)
  print(paste("Model accuracy on training set is", round(mnist.modelAccuracy(confMatrix1)*100,4), "percent."))
  
  # Accuracy on test data
  resTest<-compute(mnist.neuralnet,mnist.test$x)
  confMatrix2<-table(Expected=mnist.test$l,Predicted=((0:9)[apply(resTest$net.result-outputShift, 1, which.max)]))
  print("Confusion matrix for test set:")
  print(confMatrix2)
  print(paste("Model accuracy on test set is", round(mnist.modelAccuracy(confMatrix2)*100,4), "percent."))
  #save(mnist.neuralnet, file='nuralnet_train0.25_Thresh0.2_err1077.82_iter904437_9827_9001.RData')
}



