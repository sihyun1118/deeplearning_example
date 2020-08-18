
# Packages & options ------------------------------------------------------

options(scipen = 100)

if(!require(dplyr)) install.packages("dplyr");require(dplyr)
if(!require(ggplot2)) install.packages("ggplot2");require(ggplot2)
if(!require(reshape2)) install.packages("reshape2");require(reshape2)
if(!require(readxl)) install.packages("readxl");require(readxl)

# 100 Industry dataset
ind <- read.csv("C:/Users/sihyun/Desktop/환경부 공모전/KOSIS_data/100_Industry.csv", header = T)
colnames(ind) <- c('industry','city','region','count')
head(ind)
ind <- dcast(ind,city+region~industry); ind

# amount of using electronic
elec <- read_excel("C:/Users/sihyun/Desktop/환경부 공모전/KOSIS_data/electronic.xlsx")
head(elec)
elec <- elec[,c(2,3,4,5)]; elec
colnames(elec) <- c('city','region','type','usage');elec
tail(elec)

# basic living allowance
lm <- read.csv("C:/Users/sihyun/Desktop/환경부 공모전/KOSIS_data/기초생활수급현황(2018).csv", header = T)
lm <- lm[,c(2,3,4,5,6)]
colnames(lm) <- c('city','region','type','count_f','count'); head(lm)
