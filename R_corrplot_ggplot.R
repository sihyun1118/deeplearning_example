library(readxl)
library(dplyr)
df<-read_excel("C:/Users/sihyun/Desktop/R WD/data 천식.xlsx")
df$`SO2(ppm)`<-as.numeric(df$`SO2(ppm)`)
df$`NO2(ppm)`<-as.numeric(df$`NO2(ppm)`)
df$`CO(ppm)`<-as.numeric(df$`CO(ppm)`)
df$`PM-10(ug/m3)`<-as.numeric(df$`PM-10(ug/m3)`)
df$시도지역<-as.factor(df$시도지역)
str(df)
#시작일 지정
start_date <- as.Date("2014-01-1")
#종료일 지정
end_date <- as.Date("2018-06-30")

#월달위로 일련의 날짜 생성하기
date_set <- rep(seq(as.Date(start_date), as.Date(end_date), by = "month"), time=6)
df$날짜<-date_set
df%>%View()
if(!require(ggplot2)) install.packages("ggplot2");require(ggplot2)
if(!require(plotly)) install.packages("plotly");require(plotly)
if(!require(gridExtra)) install.packages("gridExtra");require(gridExtra)

# 기상자료와 천식 지역별 eda
library(ggplot2)
library(plotly)
library(gridExtra)
p1<-ggplot(df,aes(x=날짜,y=`월평균 발생건수(건)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p2<-ggplot(df,aes(x=날짜,y=`평균기온(°C)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p3<-ggplot(df,aes(x=날짜,y=`평균상대습도(%)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p4<-ggplot(df,aes(x=날짜,y=`평균일교차(°C)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p5<-ggplot(df,aes(x=날짜,y=`PM-10(ug/m3)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p6<-ggplot(df,aes(x=날짜,y=`SO2(ppm)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p7<-ggplot(df,aes(x=날짜,y=`NO2(ppm)`,col=시도지역))+geom_line(size=1.2)+theme_light()
p8<-ggplot(df,aes(x=날짜,y=`CO(ppm)`,col=시도지역))+geom_line(size=1.2)+theme_light()
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, nrow = 2,ncol=4)
grid.arrange(p1)

# 변수와의 관련성
if(!require(corrr)) install.packages("corrr");require(corrr)
if(!require(ggcorrplot)) install.packages("ggcorrplot");require(ggcorrplot)
if(!require(corrplot)) install.packages("corrplot");require(corrplot)
library(corrr)
library(ggcorrplot)
library(corrplot)
network_plot(cor(df[,c(6,9:15)]), legend = TRUE, min_cor=0.3)
ggcorrplot(cor(df[,c(6,9:15)]), hc.order = TRUE, type = "lower",lab = TRUE)
corrplot(cor(df[,c(6,9:15)]), method = "color",type = 'lower',order = 'hclust',addCoef.col = 'black',tl.col='black',tl.srt=45,diag = F)
corrplot.mixed(cor(df[,c(6,9:15)]), upper = "ellipse", tl.col = "black")

# 인자분석
if(!require(psych)) install.packages("psych");require(psych)
if(!require(GPArotation)) install.packages("GPArotation");require(GPArotation)
library(psych)
library(GPArotation)
df_all_weather<-df[,c(6,9:15)]
df_all_weather_factor<-principal(df_all_weather,rotate = 'none')
df_all_weather_factor$values
plot(df_all_weather_factor$values,type = 'b',ylab = "EigenValue")
abline(v=2,col='red')
df_all_weather_varimax<-principal(df_all_weather,nfactors = 2,rotate = 'varimax')
df_all_weather_varimax
biplot(df_all_weather_varimax)
df_all_weather_oblimin<-principal(df_all_weather,nfactors = 2,rotate = 'oblimin')
df_all_weather_oblimin
biplot(df_all_weather_oblimin)
KMO(cor(df[,c(6,9:15)]))
df$pc1<-df_all_weather_varimax$scores[,1]
df$pc2<-df_all_weather_varimax$scores[,2]

# 다중회귀분석
if(!require(car)) install.packages("car");require(car)
library(car)
library(plotly)
library(ggplot2)
df_model<-lm(log(`월평균 발생건수(건)`)~pc1+pc2+pc1*pc2+시도지역,data=df)
summary(df_model)
confint(df_model)
par(mfrow=c(2,2))
plot(df_model)
par(mfrow=c(1,1))
shapiro.test(df_model$residuals)
vif(df_model)
gvmodel<-gvlma(df_model)
summary(gvmodel)
df$fit<-exp(df_model$fitted.values)
ggplot(df,aes(x=pc1,y=`월평균 발생건수(건)`))+geom_point(aes(col=시도지역))+stat_smooth(method='lm',aes(y=fit,col=시도지역))
ggplot(df,aes(x=pc2,y=`월평균 발생건수(건)`))+geom_point(aes(col=시도지역))+stat_smooth(method='lm',aes(y=fit,col=시도지역))
plot_ly(x=df$pc1, y=df$pc2, z=df$`월평균 발생건수(건)`, type="scatter3d", mode="markers", color=df$시도지역)
