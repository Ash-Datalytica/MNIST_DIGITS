
#source('./R/neuralnet.R')
#debugSource('C:/Work/NNET_DIGITS/R_neuralnet_debug/R/neuralnet.r')
source('./R/compute.R')
source('./R/plot.nn.R')


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
nHidden <-3# кол-во нейронов в скрытом слое
nInput <-4# neurons count in the Input layer
nOutput <-3# neurons count in the Output layer
#Each position in the list is a matrix of weights. First position contains weights fo Input-Hidden layer
#Second position for Hidden-Output layer.
#Each matrix has one more row. Why? may be it is for biases?
startWeights <- list(matrix(data=rnorm((nInput+1)*nHidden, sd = 1.22), nrow=(nInput+1), ncol=nHidden),
                  matrix(data=rnorm((nHidden+1)*nOutput, sd = 1.22), nrow=(nHidden+1), ncol=nOutput))
nn <- neuralnet(setosa+versicolor+virginica ~ 
                  Sepal.Length+Sepal.Width
                +Petal.Length
                +Petal.Width,
                data=nnet_iristrain, 
                hidden=c(nHidden),
                lifesign="full",
                threshold=0.01,
                lifesign.step=1000,
                stepmax=1e+5,
                startweights=startWeights)

h=nn$history[[1]]#rep=1
#View(h)

##standard plot
# plot(x=h[,1],y=h[,2],type="b", xlab="stepNo", ylab="Threshold/Error", main="Learning curve", log="y" )
# lines(h[,1], y=h[,2], col="blue")
# par(new=TRUE)
# plot(x=h[,1],y=h[,3],type="b", xlab="stepNo", ylab="Threshold/Error", main="Learning curve", log="y" )
# lines(h[,1], y=h[,3], col="red")

#install.packages('ggplot2')
library(ggplot2)
df<- data.frame(h,row.names = NULL)
names(df)<-c('step','threshold','error')
p <-ggplot(df, aes(step) )
p <- p + geom_line(aes(y=error), colour="red")
p <- p + geom_line(aes(y=threshold), colour="green")
p <- p + scale_y_log10()
#p + theme(legend.position = "bottom", legend.box = "horizontal")
#p + guides(step = "colorbar", threshold = "legend", error = "legend")
#p <- p + guides(step = guide_legend("title"), threshold = guide_legend("title"),
#           error = guide_legend("title"))
p
#p <-NULL

plot(nn)
#View(nn$net.result)
mypredict <- compute(nn, iristest[-5])$net.result
# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('setosa', 'versicolor', 'virginica')[idx]
print(table(prediction, iristest$Species))
#print(nn)

#test BLAS, openBLAS, gotoBLAS2
# size=5000
# a = matrix(rnorm(size*size), size, size)
# #b = matrix(rnorm(size*size), size, size)
# system.time(c <- a%*%a)


# m = matrix (c(1:8), ncol=2, byrow=F); m
# means = apply (m, 2, mean); means
# m = t(t(m)-means); m 
# sds=apply (m, 2, sd); sds
# m = t(t(m)/sds); m 

#install.packages('raster')
library(raster)
rst <- raster(nrows = 100, ncols = 100) #create a 100x100 raster
rst[] <- round(runif(ncell(rst))) #populate raster with values, for simplicity we round them to 0 and 1
par(mfrow=c(1,2))
plot(rst) #see what you've got so far
rst.vals <- getValues(rst) #extract values from rst object
rst.cell.vals <- which(rst.vals == 1) #see which cells are 1
coords <- xyFromCell(rst, rst.cell.vals) #get coordinates of ones
rst[rst.cell.vals] <- NA #set those raster cells that are 1 to NA (you can play with rst[!rst.cell.vals] <- NA to exclude all others)
plot(rst) #a diag plot, should have only one color