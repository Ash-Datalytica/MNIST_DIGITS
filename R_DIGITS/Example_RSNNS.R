#Пример из статьи http://beyondvalence.blogspot.ru/2014/03/neural-network-prediction-of.html
#install.packages("RSNNS")
library("RSNNS") 
# Versatile neural network package
# The Stuttgart Neural Network Simulator (SNNS) is a library containing many standard 
# implementations of neural networks. This package wraps the SNNS functionality to make 
# it available from within R. Using the RSNNS low-level interface, all of the algorithmic 
# functionality and flexibility of SNNS can be accessed. Furthermore, the package contains 
# a convenient high-level interface, so that the most common neural network topologies and
# learning algorithms integrate seamlessly into R.

#Функци подсчета точности модели на основе confusion матрицы
#Удачные предсказани = сумма диагональных элементов матрицы
#Все предсказани = сумма всех элементов матрицы
modelAccuracy = function (confMatrix) {
    return (sum(diag(confMatrix))/sum(confMatrix))
}

start<-Sys.time() 
mnist <- read.csv("../MNIST_DATA/train.csv", header=TRUE) #Считываем данные в двумерную матрицу
finish<-Sys.time(); cat("Loading MNIST completed in", as.difftime((finish-start),format ="%H:%M:%S"), "second(s)")

workProportion <- 1 #процент от общего количества строк, с которыми будем далее работать
mnist.set <- mnist[sample(1:nrow(mnist),nrow(mnist)*workProportion),1:ncol(mnist)] 
#отбираем часть данных дл обучени (ускорение)

#Превращаем цифры (Labels) в классификационный вектор дл каждой строки данных
#например, вместо цифры "0" получим вектор (1,0,0,0,0,0,0,0,0,0)
#вместо "3" получим  (0,0,0,1,0,0,0,0,0,0)
mnist.targets <- decodeClassLabels(mnist.set[,1])
#This method decodes class labels from a numerical or levels vector to a binary matrix, 
#i.e., it converts the input vector to a binary matrix.
#столбцы mnist.targets соответствуют 0:9, строки - строкам  данных mnist.set

#split and normalize
#mnist.set<-splitForTrainingAndTest(mnist.set, mnist.targets, ratio=0.3)
#Usage: splitForTrainingAndTest(x, y, ratio=0.15)
#Arguments: x=inputs, y=targets
#ratio=ratio of training and test sets (default: 15% of the data is used for testing)
#mnist.set стал массивом из 4-х элементов: inputsTrain, targetstrain, inputsTest, targetsTest
#?!В статье почему-то не отрезали от данных первый столбец с Labels!
#Исправленный вызов (без labels)
mnist.set<-splitForTrainingAndTest(mnist.set[,-1], mnist.targets, ratio=0.15) #в статье ratio=0.3

start<-Sys.time() 
#mnist.set<-normTrainingAndTestSet (mnist.set, type="0_1")
mnist.set<-normTrainingAndTestSet (mnist.set, type="0_1", dontNormTargets=TRUE)
finish<-Sys.time(); 
(finish-start)
# Normalize training and test set as obtained by splitForTrainingAndTest in the following way: 
#   The inputsTrain member is normalized using normalizeData with the parameters given in type. 
# The normalization parameters obtained during this normalization are then used to normalize the 
# inputsTest member. if dontNormTargets is not set, then the targets are normalized in the same way. 
# In classification problems, normalizing the targets normally makes no sense. 
# For regression, normalizing also the targets is usually a good idea.
#?!В статье почему-то не указали dontNormTargets=TRUE !
#View (mnist.set$inputsTrain)
#View (mnist.set$targetsTrain)

## rsnns model mlp ##
start<-Sys.time() 
mnist.model <- mlp(mnist.set$inputsTrain, mnist.set$targetsTrain, size=30, #В статье используют size=5, но тогда сеть плохо предсказывает.
                   learnFuncParams=c(0.1), 
                   maxit=100, #В статье используют 100, но после 40 итераций при использовании 0.25 данных Iterative Error почти не уменьшаетс
                   inputsTest=mnist.set$inputsTest,
                   targetsTest=mnist.set$targetsTest)
finish<-Sys.time()
(finish-start)
#Create and train a multi-layer perceptron (MLP) 
#MLP состоит из 3-х слоев: вход (inputs), внутренний слой (hidden) и выход (outputs)
#В нашем случае размеры слоев 784-size-10
# mlp((x, y, size=c(5), maxit=100, initFunc="Randomize_Weights",
#      initFuncParams=c(-0.3, 0.3), learnFunc="Std_Backpropagation",
#      learnFuncParams=c(0.2, 0), updateFunc="Topological_Order",
#      updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
#      shufflePatterns=TRUE, linOut=FALSE, inputsTest, targetsTest, ...))
#Arguments:
# x=a matrix with training inputs for the network
# y=the corresponding targets values
# size= number of units in the hidden layer(s)
# maxit= maximum of iterations to learn
# initFunc= the initialization function to use
# initFuncParams= the parameters for the initialization function
# learnFunc=the learning function to use
# learnFuncParams= the parameters for the learning function
# updateFunc = the update function to use
# updateFuncParams =the parameters for the update function
# hiddenActFunc =the activation function of all hidden units
# shufflePatterns =should the patterns be shuffled?
# linOut =sets the activation function of the output units to linear or logistic
# inputsTest =a matrix with inputs to test the network
# targetsTest =the corresponding targets for the test input
# ... = additional function parameters (currently not used) 


#Examining the results#
#summary (mnist.model)
#mnist.model
#weightMatrix(mnist.model) #weights for all neurons
#extractNetInfo(mnist.model)

#confusion matrix, train
train.con <- confusionMatrix(mnist.set$targetsTrain, mnist.model$fitted.values )
dimnames (train.con)$targets <- c(0:9)
dimnames (train.con)$predictions <- c(0:9)
train.con
modelAccuracy(train.con) #Точность модели на обучающих данных
#Note that R will mask the confusionMatrix() function from the caret package if you load RSNNS after caret- 
#access it using caret::confusionMatrix()).
#мы используем функцию из RSNNS


#confusion matrix, test
test.con <- confusionMatrix(mnist.set$targetsTest, mnist.model$fittedTestValues )
dimnames (test.con)$targets <- c(0:9); dimnames (test.con)$predictions <- c(0:9)
test.con
modelAccuracy(test.con)#Точность модели на тестовых данных

# Iterative Error
#For our first visualization, we can plot the sum of squared errors for each iteration of the model 
#for both the training and test sets. RSNNS has a function called plotIterativeError() which will allow us
#to see the progression of the neural network training.
plotIterativeError(mnist.model, main="Iterative Error mnist.model")
legend("topright", c("fitted values", "fitted test values"),
      col=c("black","red"), lwd=c(1,1))

