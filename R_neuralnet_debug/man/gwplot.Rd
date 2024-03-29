\name{gwplot}
\alias{gwplot}
\title{ Plot method for generalized weights }
\description{
  \code{gwplot}, a method for objects of class \code{nn}, typically
  produced by \code{neuralnet}. 
Plots the generalized weights (Intrator and Intrator, 1993) for one specific covariate and one response variable.
}
\usage{
gwplot(x, rep = NULL, max = NULL, min = NULL, file = NULL, 
       selected.covariate = 1, selected.response = 1, 
       highlight = FALSE, type="p", col = "black", \dots)
}
\arguments{
  \item{x}{ an object of class \code{nn} }
  \item{rep}{ an integer indicating the repetition to plot. If rep="best", the repetition with the smallest error will be plotted. If not stated all repetitions will be plotted. }
  \item{max}{ maximum of the y axis. In default, max is set to the highest y-value. }
  \item{min}{ minimum of the y axis. In default, min is set to the smallest y-value. }
  \item{file}{ a character string naming the plot to write to. If not stated, the plot will not be saved. }
  \item{selected.covariate}{ either a string of the covariate's name or an integer of the ordered covariates, indicating the reference covariate in the generalized weights plot. Defaulting to the first covariate. }
  \item{selected.response}{ either a string of the response variable's name or an integer of the ordered response variables, indicating the reference response in the generalized weights plot. Defaulting to the first response variable. }
  \item{highlight}{ a logical value, indicating whether to highlight (red color) the best repetition (smallest error). Only reasonable if rep=NULL. Default is FALSE}
  \item{type}{a character indicating the type of plotting; actually any of the types as in \code{\link{plot.default}}.}
  \item{col}{ a color of the generalized weights.  }
  \item{\dots}{ Arguments to be passed to methods, such as graphical parameters (see \code{\link{par}}). }
}

\references{ 
    Intrator O. and Intrator N. (1993)
    \emph{Using Neural Nets for Interpretation of Nonlinear Models.}  
    Proceedings of the Statistical Computing Section, 244-249 
    San Francisco: American Statistical Society (eds.) }
\author{ Stefan Fritsch, Frauke Guenther \email{guenther@bips.uni-bremen.de} }
\seealso{\code{\link{neuralnet}}}
\examples{
data(infert, package="datasets")
print(net.infert <- neuralnet(case~parity+induced+spontaneous, infert, 
		                err.fct="ce", linear.output=FALSE, likelihood=TRUE))
gwplot(net.infert, selected.covariate="parity")
gwplot(net.infert, selected.covariate="induced")
gwplot(net.infert, selected.covariate="spontaneous")
}
\keyword{ neural }
