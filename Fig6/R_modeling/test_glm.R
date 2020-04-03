
require('lme4')
require('nlme')
require('lmerTest')
require('pbkrtest')

PFC = 1
VERTEX = 0
data <- read.csv(file="data_for_R_model.csv", sep="\t")
data_raw <- read.csv(file="data_for_R_model_raw_TMS.csv", sep="\t")



data[data$trial <= 225,"session_half"] = 0
data[data$trial > 225,"session_half"] = 1


# Model 1: weak evidence that serial biases decrease through time
model1='err ~ session_half*sin(prev_curr)*tms_intensity+ (1+sin(prev_curr)|subject)'
m1=lmerTest::lmer(model1,data[data$location == 1,])
m1=lme(err ~ session_half*tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data[data$location == 1,])

summary(m1)
coefs1 <- data.frame(coef(summary(m1)))
coefs1$p.z <- 2 * (1 - pnorm(abs(coefs1$t.value)))
coefs1

# Model 1: weak evidence that serial biases decrease through time, separated by exp
model1_exp='err ~ session_half*sin(prev_curr)*tms_intensity+ (1+sin(prev_curr)|subject)'
m1_exp0=lme(err ~ session_half*tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data[data$location == 1 &  data$study==0,])
m1_exp1=lme(err ~ session_half*tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data[data$location == 1 &  data$study==1,])

coefs1 <- data.frame(coef(summary(m1_exp0)))
coefs1 <- data.frame(coef(summary(m1_exp1)))
coefs1$p.z <- 2 * (1 - pnorm(abs(coefs1$t.value)))
coefs1


# Model 2: TMS affects PFC, but not Vertex
model2='err ~  location*tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m2=lme(err ~ location*tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data)
#m2=lmerTest::lmer(model2,data)
summary(m2)
coefs2 <- data.frame(coef(summary(m2)))
coefs2$p.z <- 2 * (1 - pnorm(abs(coefs2$t.value)))
coefs2

# r <- function(data, indices){
# 	d <- data[indices,]
# 	m2=lme(err ~ location*tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=d)
# 	coefs3 <- data.frame(coef(summary(m2)))
# 	return(coefs3$Estimate[4])
# }

# results <- boot(data=data, statistic=r, R=1000)




# Model 3: TMS affects PFC
model3='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m3=lmerTest::lmer(model3,data[data$location == 1,])
m3=lme(err ~  tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data[data$location == 1,])
summary(m3)
coefs3 <- data.frame(coef(summary(m3)))
coefs3$p.z <- 2 * (1 - pnorm(abs(coefs3$t.value)))
coefs3


# Model 4: TMS does not affect Vertex
model4='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m4=lmerTest::lmer(model4,data[data$location == 0,])
m4=lme(err ~  tms_intensity*sin(prev_curr), random= ~1+sin(prev_curr)|subject, data=data[data$location == 0,])
summary(m4)
coefs4 <- data.frame(coef(summary(m4)))
coefs4$p.z <- 2 * (1 - pnorm(abs(coefs4$t.value)))
coefs4

# Model 5: early trials: TMS affects PFC, but not Vertex
model5='err ~  location*tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m5=lmer(model5,data[data$trial < 225,])
m5 = lme(err ~  location*tms_intensity*sin(prev_curr), random=~1+sin(prev_curr)|subject,data=data[data$trial < 225,])
summary(m5)
coefs5 <- data.frame(coef(summary(m5)))
coefs5$p.z <- 2 * (1 - pnorm(abs(coefs5$t.value)))
coefs5



# Model 6: early trials: TMS affects PFC
model7='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m7=lmer(model7,data[data$location == 1 & data$trial < 225 ,])
m7 = lme(err ~  tms_intensity*sin(prev_curr), random=~1+sin(prev_curr)|subject,data=data[data$location == 1 & data$trial < 225 ,])
summary(m7)
coefs7 <- data.frame(coef(summary(m7)))
coefs7$p.z <- 2 * (1 - pnorm(abs(coefs7$t.value)))
coefs7

# Model 7: early trials: TMS does not affect Vertex
model7='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m7=lmer(model7,data[data$location == 0 & data$trial < 225,])
m7 = lme(err ~  tms_intensity*sin(prev_curr), random=~1+sin(prev_curr)|subject,data=data[data$location == 0 & data$trial < 225 ,])

summary(m7)
coefs7 <- data.frame(coef(summary(m7)))
coefs7$p.z <- 2 * (1 - pnorm(abs(coefs7$t.value)))
coefs7

# STUDY 1 vs STUDY 2
# Model 8: TMS affects PFC
model8='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m8=lmer(model8,data[data$location == 1 & data$trial < 225 & data$study==0,])
m8 = lme(err ~  tms_intensity*sin(prev_curr), random=~1+sin(prev_curr)|subject,data=data[data$location == 1 & data$trial < 225 & data$study==0,])

summary(m8)
coefs8 <- data.frame(coef(summary(m8)))
coefs8$p.z <- 2 * (1 - pnorm(abs(coefs8$t.value)))
coefs8

model9='err ~  tms_intensity*sin(prev_curr) + (1+sin(prev_curr)|subject)'
m9=lmer(model9,data[data$location == 1 & data$trial < 225 & data$study==1,])
m9 = lme(err ~  tms_intensity*sin(prev_curr), random=~1+sin(prev_curr)|subject,data=data[data$location == 1 & data$trial < 225 & data$study==1,])

summary(m9)
coefs9 <- data.frame(coef(summary(m9)))
coefs9$p.z <- 2 * (1 - pnorm(abs(coefs9$t.value)))
coefs9

##### ERROR TESTING ####

data <- read.csv(file="data_for_R_model_raw_tms.csv", sep="\t")

data <- read.csv(file="data_for_R_model.csv", sep="\t")

# data$tms_intensity = factor(data$tms_intensity)


model7='err**2 ~  tms_intensity*location + (1+sin(prev_curr)|subject)'
m7=lmer(model7,data)
summary(m7)
coefs7 <- data.frame(coef(summary(m7)))
coefs7$p.z <- 2 * (1 - pnorm(abs(coefs7$t.value)))
coefs7

model8='err**2 ~  tms_intensity + (1+sin(prev_curr)|subject)'
m8=lmer(model8,data[data$location == PFC,])
summary(m8)
coefs8 <- data.frame(coef(summary(m8)))
coefs8$p.z <- 2 * (1 - pnorm(abs(coefs8$t.value)))
coefs8


model9='err**2 ~  tms_intensity + (1+sin(prev_curr)|subject)'
m9=lmer(model9,data[data$location == VERTEX,])
summary(m9)
coefs9 <- data.frame(coef(summary(m9)))
coefs9$p.z <- 2 * (1 - pnorm(abs(coefs9$t.value)))
coefs9