df1 = read.csv("/media/petrichor/data/future/autoencoders/visualizations/weights/ae.csv")
df2 = read.csv("/media/petrichor/data/future/autoencoders/visualizations/weights/vaeout.csv")

library(ggplot2)
library(gridExtra)

df2$labels<- as.factor(df2$labels)
g1 <-ggplot(df2, aes(x, y=y, color = df2$labels)) +
  geom_point(size = 3 ,alpha=.5) +
  ggtitle("Variational autoencoder")

df1$labels<- as.factor(df1$labels)
g2 <-ggplot(df1, aes(x, y=y, color = df1$labels)) +
  geom_point(size = 3 ,alpha=.5) +
  ggtitle("Standard Autoencoder")

gg=arrangeGrob(g1, g2, ncol=1)
ggsave("/media/petrichor/data/future/autoencoders/visualizations/ggplot.jpg",gg)