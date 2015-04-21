library ('neuralnet')
library ('RSNNS')
source('MNISTLoader.R')

start<-Sys.time() 
mnist.loadFromBin('../MNIST_DATA/UNZIP/') #60000+10000 цифр, 4-5 сек
finish<-Sys.time() ; print("Data loaded"); print (finish-start)


##дл отладки оставлем часть данных дл обучени (ускорение)
mnist.makeSample(trainRatio = 0.005,testRatio=0.5)
print(paste('Training sample size is',mnist.train$n, "rows."))

mnist.normalize() #normalize train and test data
#   mnist.train$x<-RSNNS::normalizeData(mnist.train$x, type = '0_1')
#   mnist.test$x<-RSNNS::normalizeData(mnist.test$x, type = '0_1')
#   #Set colnames for all used matrices
#   colnames(mnist.train$x)<-paste0('x',1:ncol(mnist.train$x))
#   colnames(mnist.test$x)<-paste0('x',1:ncol(mnist.test$x))
#View (mnist.train$x)

#Формула в формате neuralnet (кто от кого зависит)
#formula=y0+...y9 ~ x1+...+x784
myFormula = paste(paste(colnames(mnist.train$y),collapse='+'),'~',
                  paste(colnames(mnist.train$x),collapse='+'))

#debugonce(neuralnet)
#mnist.train$y<-mnist.train$y==1

start<-Sys.time()
#Сеть должна закончить работу, достигнув threshold, иначе она не посчитает матрицу вестов и т.п.
mnist.neuralnet <- neuralnet (formula=myFormula, 
                              data=cbind(mnist.train$y,mnist.train$x), 
                              hidden = (30), 
                              #learningrate=0.1, #a numeric value specifying the learning rate used by 
                              # traditional backpropagation. 
                              #Used ONLY for traditional backpropagation.
                              algorithm='slr', # 'backprop', 'rprop+'(default), 'rprop-', 'sag', or 'slr'.
                              rep=1,
                              threshold = 0.1, #an integer specifying the threshold for the partial derivatives of the error function  as stopping criteria. Default: 0.01.
                              act.fct = "tanh", #'logistic' (default) or 'tanh'
                              err.fct="sse", #"sse"=’sum of squared errors’ and "ce"=’cross entropy’.
                              linear.output=FALSE,#If act.fct should not be applied to the output neurons set linear output to TRUE (default), otherwise to FALSE.
                              stepmax=250000, #the maximum steps for the training of the neural network. Reaching this maximum leads to a stop of the neural network's training process.
                              lifesign="full", 
                              lifesign.step=100
)
finish<-Sys.time(); print('NNet training completed in'); print((finish-start))

print(mnist.neuralnet)
#res=mnist.neuralnet$net.result  
#names(res)="data"
#View(data.frame(Label=mnist.train$l, Predicted=res, Expected=mnist.train$y))
##View(res$data)
##sum(res$data[1,])
#mnist.error<-mnist.neuralnet$result.matrix[1]
#mnist.reached.threshold<-mnist.neuralnet$result.matrix[2]
mnist.steps<-mnist.neuralnet$result.matrix[3]
print (paste("Скорость расчета сети:",round(mnist.steps/as.numeric((finish-start), units="mins")),
             "итераций/мин"))  

# Accuracy on training data
resTrain<-compute(mnist.neuralnet,mnist.train$x)
#View(resTrain$net.result)
# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
predictedLbl <- apply(resTrain$net.result, c(1), maxidx)-1 #predicted labels
#View (cbind(predictedLbl,mnist.train$l))
confMatrix1<-table(Expected=mnist.train$l,Predicted=predictedLbl)
print("Confusion matrix for training set:")
print(confMatrix1)
print(paste("Model accuracy on training set is", round(mnist.modelAccuracy(confMatrix1)*100,4), "percent."))

# Accuracy on test data
resTest<-compute(mnist.neuralnet,mnist.test$x)
confMatrix2<-table(Expected=mnist.test$l,Predicted=(apply(resTest$net.result, c(1), maxidx)-1))
print("Confusion matrix for test set:")
print(confMatrix2)
print(paste("Model accuracy on test set is", round(mnist.modelAccuracy(confMatrix2)*100,4), "percent."))

