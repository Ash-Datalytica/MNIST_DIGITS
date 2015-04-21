library(neuralnet)
library (datasets)

##Example 1
# data(infert, package="datasets")
# #View(infert)
# print(net.infert <- neuralnet(case~parity+induced+spontaneous+age, infert,
#                               err.fct="ce", linear.output=FALSE, likelihood=TRUE))
# gwplot(net.infert, selected.covariate="parity")
# gwplot(net.infert, selected.covariate="induced")
# gwplot(net.infert, selected.covariate="spontaneous")


##Example 2
# Var1 <- runif(50, 0, 100) 
# sqrt.data <- data.frame(Var1, Sqrt=sqrt(Var1))
# print(net.sqrt <- neuralnet(Sqrt~Var1,  sqrt.data, hidden=10, 
#                           threshold=0.01))
# #net.sqrt$response
# compute(net.sqrt, (1:10)^2)$net.result #отгаданные значани функции (y) на основе входных данных (x)

# #Example 3
# Var1 <- rpois(100,0.5)
# Var2 <- rbinom(100,2,0.6)
# Var3 <- rbinom(100,1,0.5)
# SUM <- as.integer(abs(Var1+Var2+Var3+(rnorm(100))))
# sum.data <- data.frame(Var1+Var2+Var3, SUM)
# print(net.sum <- neuralnet( SUM~Var1+Var2+Var3,  sum.data, hidden=1, 
#                             act.fct="tanh"))
# 
# net.sum$net.result
# #cbind(sum.data,net.sum$net.result)
# main <- glm(SUM~Var1+Var2+Var3, sum.data, family=poisson())
# full <- glm(SUM~Var1*Var2*Var3, sum.data, family=poisson())
# prediction(net.sum, list.glm=list(main=main, full=full))

##Example 4
#http://horicky.blogspot.ru/2012/06/predictive-analytics-neuralnet-bayesian.html


# Prepare training and testing data
testidx <- which(1:length(iris[,1])%%5 == 0)
iristrain <- iris[-testidx,]
iristest <- iris[testidx,]
nnet_iristrain <-iristrain
#Binarize the categorical output
nnet_iristrain <- cbind(nnet_iristrain, 
                        iristrain$Species == 'setosa')
nnet_iristrain <- cbind(nnet_iristrain,
                        iristrain$Species == 'versicolor')
nnet_iristrain <- cbind(nnet_iristrain, 
                        iristrain$Species == 'virginica')
names(nnet_iristrain)[6] <- 'setosa'
names(nnet_iristrain)[7] <- 'versicolor'
names(nnet_iristrain)[8] <- 'virginica'
#View(nnet_iristrain)
nn <- neuralnet(setosa+versicolor+virginica ~ 
                  Sepal.Length+Sepal.Width
                +Petal.Length
                +Petal.Width,
                data=nnet_iristrain, 
                hidden=c(3),
                lifesign="full",
                lifesign.step=1000)
plot(nn)
View(nn$net.result)
mypredict <- compute(nn, iristest[-5])$net.result
# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('setosa', 'versicolor', 'virginica')[idx]
print(table(prediction, iristest$Species))
print(nn)
