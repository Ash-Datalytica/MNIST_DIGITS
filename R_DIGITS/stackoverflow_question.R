#################### Importing the data into R ##########
#path <- "path_to_data_folder/MNIST_database_of_handwritten_digits/"  # Data can be downloaded from: http://yann.lecun.com/exdb/mnist/
path <- "../MNIST_DATA/UNZIP/"
to.read <- file(paste0(path, "train-images.idx3-ubyte"), "rb")
to.read_Label <- file(paste0(path, "train-labels.idx1-ubyte"), "rb")
magicNumber <- readBin(to.read, integer(), n=1, endian="big")
magicNumber_Label <- readBin(to.read_Label, integer(), n=1, endian="big")
numberOfImages <- readBin(to.read, integer(), n=1, endian="big")
numberOfImages_Label <- readBin(to.read_Label, integer(), n=1, endian="big")
rowPixels <- readBin(to.read, integer(), n=1, endian="big")
columnPixels <- readBin(to.read, integer(), n=1, endian="big")

trainDigits <- NULL

#Trick #1: read unsigned data
trainDigits <- replicate(numberOfImages,c(matrix(readBin(to.read, integer(), n=(rowPixels*columnPixels), 
                                                         size=1, endian="big", signed=F),
                                                 rowPixels,columnPixels)[,columnPixels:1]))
trainDigits <- data.frame(t(trainDigits),row.names=NULL)
trainDigits_Label <- replicate(numberOfImages,readBin(to.read_Label, integer(), n=1, size=1, 
                                                      endian="big", signed=F))
close(to.read)
close(to.read_Label)

#################### Test Data ####################

to.read_test <- file(paste0(path, "t10k-images.idx3-ubyte"), "rb")
to.read_Label_test <- file(paste0(path, "t10k-labels.idx1-ubyte"), "rb")
magicNumber <- readBin(to.read_test, integer(), n=1, endian="big")
magicNumber_Label <- readBin(to.read_Label_test, integer(), n=1, endian="big")
numberOfImages_test <- readBin(to.read_test, integer(), n=1, endian="big")
numberOfImages_Label_test <- readBin(to.read_Label_test, integer(), n=1, endian="big")
rowPixels <- readBin(to.read_test, integer(), n=1, endian="big")
columnPixels <- readBin(to.read_test, integer(), n=1, endian="big")

#read unsigned data 
testDigits <- replicate(numberOfImages_test,c(matrix(readBin(to.read, integer(), n=(rowPixels*columnPixels), 
                                                             size=1, endian="big", signed=F),
                                                     rowPixels,columnPixels)[,columnPixels:1]))
testDigits <- data.frame(t(testDigits),row.names=NULL)
testDigits_Label <- replicate(numberOfImages_test,readBin(to.read_Label_test, integer(), n=1, size=1, 
                                                          endian="big", signed=F))
close(to.read_test)
close(to.read_Label_test)

#################### Modelling ####################

library(neuralnet)

#add Label data to training data.frame
trainData <- cbind(trainDigits_Label, trainDigits)
names(trainData)[1] <- "Label"

#Reduce training data for speedup
trainSample <- 1000 #use more then 500 rows to get better model accuracy (slow!)
trainData <- trainData[1:trainSample,]
myThreshold <- trainSample/5000 #use smaller threshold to get better model accuracy (slow!)

#Trick #2: normalize and center pixel data before trainig and testing
normFactor <- max(trainData) #=255
trainData[,-1] <- trainData[,-1]/normFactor #normalize inputs
centerFactor <- mean(as.matrix(trainData[,-1])) #0.5 mean по столбцу? 
trainData[,-1] <- trainData[,-1]- centerFactor #center inputs
testDigits <- testDigits/normFactor - centerFactor

#Trick #3: use more neurons in the hidden layer to rise the model accuracy
nHidden=30

#train model which predicts Labels
myFormula <- as.formula(paste0("Label ~ ", paste0("X",1:(ncol(trainDigits)), collapse="+")))
myNnet <- neuralnet(formula = myFormula, data = trainData, hidden = c(nHidden), 
                    algorithm='rprop+', #learningrate=0.01,
                    learningrate.limit=list(min=c(1e-10), max=c(0.01)), #default values min/max = 1e-10/0.1
                    learningrate.factor=list(minus=c(0.5), plus=c(1.2)), #default values minus/plus = 0.5/1.2
                    err.fct="sse", #Using "sum square errors" function for Error
                    act.fct="tanh",#Using tangent hyperbolicus activation smoothing function 
                    threshold=myThreshold,
                    lifesign="full", lifesign.step=500,
                    stepmax=3e05)

#Trick #4: get rid of negative predictions. consider them to be equal to zero. 
#The same with too big predictions (>9)
myNnet$net.result[[1]][myNnet$net.result[[1]]<0]<-0
myNnet$net.result[[1]][myNnet$net.result[[1]]>9]<-9

#################### 'neuralnet' Predictions ####################

predictOut <- compute(myNnet, testDigits)
predictOut$net.result[predictOut$net.result<0] <- 0
predictOut$net.result[predictOut$net.result>9] <- 9

#################### Result analysis ####################

#Model accuracy on training data
confTrain <- table(Predicted=round(myNnet$net.result[[1]]), Expected=(trainData[,"Label"]))
print("NN to predict Labels.")
print("Confusion matrix for training set:")
print (confTrain)
print(paste0("Model accuracy on training set is ", round(sum(diag(confTrain))/sum(confTrain)*100,4), "%"))

#Model accuracy on test data
confTest <- table(Predicted=round(predictOut$net.result), Expected=testDigits_Label)
print("Confusion matrix for test set:")
print (confTest)
print(paste0("Model accuracy on test set is ", round(sum(diag(confTest))/sum(confTest)*100,4), "%"))



#########################################################################################
#Trick #5: Predict digit Class instead of predicting digit Label
#Replace each Label with a vector of 10 bits "Label classes"
library (nnet)

# appending the Label classes to the training data
output <- nnet::class.ind(trainData[,"Label"])
colnames(output)<-paste0('out.',colnames(output))
output.names<-colnames(output)
input.names<-colnames(trainData[,-1])
trainData <-cbind(output,trainData)

#train model which predicts Label classes
myFormula <- as.formula(paste0(paste0(output.names,collapse='+')," ~ ", 
                               paste0(input.names, collapse="+")))
myNnetClass <- neuralnet(formula = myFormula, data = trainData, hidden = c(nHidden), 
                  algorithm='sag', #learningrate=0.01,
                  learningrate.limit=list(min=c(1e-10), max=c(0.01)), #default values min/max = 1e-10/0.1
                  learningrate.factor=list(minus=c(0.5), plus=c(1.2)), #default values minus/plus = 0.5/1.2
                  err.fct="sse", #Using "sum square errors" function for Error
                  act.fct="tanh",#Using tangent hyperbolicus activation smoothing function 
                  threshold=myThreshold, 
                  lifesign="full", lifesign.step=500,
                  stepmax=3e05)


# Convert  binary output to categorical output (labels)
nnres=myNnetClass$net.result[[1]]
myNnetClass$net.result[[1]] <- (0:9)[apply(myNnetClass$net.result[[1]],1,which.max)]


#################### 'neuralnet' Predictions ####################

predictOutClass <- compute(myNnetClass, testDigits)
colnames(predictOutClass$net.result) <- paste0("Cl", 0:9)
predictedLabel <- (0:9)[apply(predictOutClass$net.result, 1, which.max)]

#################### Result analysis ####################

#Model accuracy on training data
confTrain <- table(Predicted=myNnetClass$net.result[[1]], Expected=trainData[,"Label"])
print("NN to predict Label Classes.")
print("Confusion matrix for training set:")
print (confTrain)
print(paste0("Model accuracy on training set is ", round(sum(diag(confTrain))/sum(confTrain)*100,4), "%"))

#Model accuracy on test data
confTest <- table(Predicted=predictedLabel, Expected=testDigits_Label)
print("Confusion matrix for test set:")
print (confTest)
print(paste0("Model accuracy on test set is ", round(sum(diag(confTest))/sum(confTest)*100,4), "%"))

