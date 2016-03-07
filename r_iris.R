library(OpenML)
library(ggplot2)
library(cowplot)
library(mlr)
library(graphics)
library(rpart)
library(party)
# Helper function for visualizing the hyperrectangles

library("grid")
library("coin")
Colors <- colorspace::rainbow_hcl(3)
Colors_trans <- apply(rbind(col2rgb(colorspace::rainbow_hcl(3)), alpha = 100, maxColorValue = 255), 2, 
                      function(x) do.call("rgb", as.list(x)))
plot_rectangles <- function(obj, x, y, class, depth) {
  xname <- paste(deparse(substitute(x), 500), collapse = "\n")
  yname <- paste(deparse(substitute(y), 500), collapse = "\n")
  grid.newpage()
  pushViewport(plotViewport(c(5, 4, 2, 2)))
  pushViewport(dataViewport(x,
                            y,
                            name="plotRegion"))
  grid.points(x, y, pch = 19,
              gp=gpar(cex=0.5, col = Colors[class]))
  grid.rect()
  grid.xaxis()
  grid.yaxis()
  grid.text(xname, y=unit(-3, "lines"))
  grid.text(yname, x=unit(-3, "lines"), rot=90)
  seekViewport("plotRegion")
  plot_rect(obj@tree, xname = xname, depth)
  grid.points(x, y, pch = 19,
              gp=gpar(cex=0.5, col = Colors[class]))
}

plot_rect <- function(obj, xname, depth) {
  if (!missing(depth)) {
    if (obj$nodeID >= depth) return()
  }
  if (obj$psplit$variableName == xname) {
    x <- unit(rep(obj$psplit$splitpoint, 2), "native")
    y <- unit(c(0, 1), "npc")
  } else {
    x <- unit(c(0, 1), "npc")
    y <- unit(rep(obj$psplit$splitpoint, 2), "native")
  }
  grid.lines(x, y)
  if (obj$psplit$variableName == xname) {
    pushViewport(viewport(x = unit(current.viewport()$xscale[1], "native"),
                          width = x[1] - unit(current.viewport()$xscale[1], "native"),
                          xscale = c(unit(current.viewport()$xscale[1], "native"), x[1]),
                          yscale = current.viewport()$yscale,
                          just = c("left", "center")))
  } else {
    pushViewport(viewport(y = unit(current.viewport()$yscale[1], "native"),
                          height = y[1] - unit(current.viewport()$yscale[1], "native"),
                          xscale = current.viewport()$xscale,
                          yscale = c(unit(current.viewport()$yscale[1], "native"), y[1]),
                          just = c("center", "bottom")))
  }
  pred <- ifelse(length(obj$left$prediction) == 1, as.integer(obj$left$prediction > 0.5) + 1, which.max(obj$left$prediction))
  grid.rect(gp = gpar(fill = "white"))
  grid.rect(gp = gpar(fill = Colors_trans[pred]))
  if (!is(obj$left, "TerminalNode")) {
    plot_rect(obj$left, xname, depth)
  } 
  popViewport()
  if (obj$psplit$variableName == xname) {
    pushViewport(viewport(x = unit(x[1], "native"),
                          width = unit(current.viewport()$xscale[2], "native")-x[1],
                          xscale = c(x[1], unit(current.viewport()$xscale[2], "native")),
                          yscale = current.viewport()$yscale,
                          just = c("left", "center")))
  } else {
    pushViewport(viewport(y = unit(y[1], "native"),
                          height = unit(current.viewport()$yscale[2], "native")-y[1],
                          xscale = current.viewport()$xscale,
                          yscale = c(y[1], unit(current.viewport()$yscale[2], "native")),
                          just = c("center", "bottom")))
  }
  pred <- ifelse(length(obj$right$prediction) == 1, as.integer(obj$right$prediction > 0.5) + 1, which.max(obj$right$prediction)) 
  grid.rect(gp = gpar(fill = "white"))
  grid.rect(gp = gpar(fill = Colors_trans[pred]))
  if (!is(obj$right, "TerminalNode")) { 
    plot_rect(obj$right, xname, depth)
  } 
  popViewport()
}
iris = getOMLDataSet(did = 61, verbosity=0)$data
iristask = makeClassifTask(data = iris, target = "class")
head(iris)
ggplot(data=iris, aes(x=sepallength, y=sepalwidth, color=class, shape=class)) + geom_point(size=2) + theme_cowplot() + theme(legend.position = c(0.14, 0.80))

tree_iris <- ctree(class ~ ., data = iris)
with(iris, plot_rectangles(tree_iris, petallength, petalwidth, class, 2))