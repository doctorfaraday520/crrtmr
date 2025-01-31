library(tidyverse)
library(grf)        
library(MatchIt)    
library(gridExtra)  
library(survival)  
library(mice)       
library(keras)
library(tensorflow)
library(dplyr)
library(tidyr)
library(readr)
library(gbmt)
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)
library(ggpubr)
rm(list =ls())


set.seed(42)
# setwd('matterarising/')
generate_covariates <- function(n) {
  age <- round(rnorm(n, mean = 63, sd = 15))   
  male <- rbinom(n, 1, 0.61)                   
  ethnicity <- rbinom(n, 1, 0.30)              
  race <- sample(1:7, n, replace = TRUE, 
                 prob = c(0.004, 0.08, 0.12, 0.02, 0.005, 0.25, 0.52))
  # 1=American Indian, 2=Asian, 3=Black, 4=Multiple, 
  # 5=Pacific Islander, 6=Unknown, 7=White
  heart_comorb <- rbinom(n, 1, 0.75)
  liver_comorb <- rbinom(n, 1, 0.65)           
  infection_comorb <- rbinom(n, 1, 0.60)       
  other_comorb <- rbinom(n, 1, 0.05)          
  prev_crrt <- rpois(n, 0.2)                   
  creatinine <- rnorm(n, mean = 3.5, sd = 1.5)  
  bun <- rnorm(n, mean = 60, sd = 25)           
  map <- rnorm(n, mean = 65, sd = 10)           
  heart_rate <- rnorm(n, mean = 95, sd = 20)    
  tbil <- exp(rnorm(n, log(2), 0.7) )         
  inr <- rnorm(n, mean = 1.5, sd = 0.4)        
  wbc <- exp(rnorm(n, log(12), 0.5) )         
  pao2_fio2 <- rnorm(n, mean = 200, sd = 80)   
  data.frame(
    age = age,
    male = male,
    ethnicity = ethnicity,
    race = race,
    heart_comorb = heart_comorb,
    liver_comorb = liver_comorb,
    infection_comorb = infection_comorb,
    other_comorb = other_comorb,
    prev_crrt = prev_crrt,
    creatinine = creatinine,
    bun = bun,
    map = map,
    heart_rate = heart_rate,
    tbil = tbil,
    inr = inr,
    wbc = wbc,
    pao2_fio2 = pao2_fio2
  )
}
clean_covariates <- function(data) {
  data$age <- pmax(pmin(data$age, 100), 18)
  data$male <- factor(data$male, 
                      levels = c(0,1), 
                      labels = c("Female", "Male"))
  data$ethnicity <- factor(data$ethnicity,
                           levels = c(0,1),
                           labels = c("Non-Hispanic", "Hispanic"))
  
  data$race <- factor(data$race,
                      levels = 1:7,
                      labels = c("American Indian", "Asian", "Black", 
                                 "Multiple", "Pacific Islander", 
                                 "Unknown", "White"))

  comorbidity_vars <- c("heart_comorb", "liver_comorb", 
                        "infection_comorb", "other_comorb")
  for(var in comorbidity_vars) {
    data[[var]] <- factor(data[[var]], 
                          levels = c(0,1), 
                          labels = c("No", "Yes"))
  }
  
  data$prev_crrt <- pmin(pmax(data$prev_crrt, 0), 10)
  data$creatinine <- pmax(pmin(data$creatinine, 20), 0.2)
  data$bun <- pmax(pmin(data$bun, 200), 2)
  data$map <- pmax(pmin(data$map, 200), 30)
  data$heart_rate <- pmax(pmin(data$heart_rate, 200), 30)
  data$tbil <- pmax(pmin(data$tbil, 60), 0.1)
  data$inr <- pmax(pmin(data$inr, 10), 0.5)
  data$wbc <- pmax(pmin(data$wbc, 100), 0.1)
  data$pao2_fio2 <- pmax(pmin(data$pao2_fio2, 300), 100)
  data <- data %>%
    mutate(across(where(is.numeric), ~ifelse(is.infinite(.) | is.na(.), 
                                             median(.[!is.infinite(.) &!is.na(.)]), 
                                             .)))
  
  attr(data$age, "label") <- "Age (years)"
  attr(data$creatinine, "label") <- "Creatinine (mg/dL)"
  attr(data$bun, "label") <- "Blood Urea Nitrogen (mg/dL)"
  attr(data$map, "label") <- "Mean Arterial Pressure (mmHg)"
  attr(data$heart_rate, "label") <- "Heart Rate (bpm)"
  attr(data$tbil, "label") <- "Total Bilirubin (mg/dL)"
  attr(data$inr, "label") <- "International Normalized Ratio"
  attr(data$wbc, "label") <- "White Blood Cell Count (K/uL)"
  attr(data$pao2_fio2, "label") <- "PaO2/FiO2 Ratio"
  attr(data$prev_crrt, "label") <- "Previous CRRT Episodes"
  
  return(data)
}

# Beneficial CRRT effect
coefficients <- c(
  intercept = 6,
  treatmentCRRT = 1,      
  age = -0.002,
  maleMale = 0.02,
  ethnicityHispanic = 0.001,
  raceAsian = 0.01,
  raceBlack = 0.01,
  raceMultiple = 0.01,
  racePacific.Islander = 0.01,
  raceWhite =0.01,
  heart_comorbYes = -0.05,
  liver_comorbYes = -0.05,
  infection_comorbYes = -0.08,
  other_comorbYes = -0.02,
  prev_crrt = -0.03,
  creatinine = -0.08,
  bun = -0.008,
  map = 0.008,
  heart_rate = -0.08,
  tbil = -0.08,
  inr = -0.08,
  wbc = -0.008,
  pao2_fio2 = 0.008
)

invlogit <- function(x) {
  1 / (1 + exp(-x))
}

patient_data <- generate_covariates(10000) %>% clean_covariates()
patient_data$treatment <- rbinom(10000, 1, 0.5)
patient_data$treatment <- ifelse(patient_data$treatment == 1, 'CRRT', 'No_CRRT')
patient_data$treatment <- factor(patient_data$treatment, levels = c("No_CRRT", "CRRT"))
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                         coefficients["treatmentCRRT"] * (treatment == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability <- invlogit(linear_predictor)
patient_data$outcomes <- ifelse(patient_data$outcome_probability >= 0.5, 'Positive', 'Negative')

patient_data$treatment_counterfactual <- ifelse(patient_data$treatment == 'No_CRRT', "CRRT", 'No_CRRT')
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                         coefficients["treatmentCRRT"] * (treatment_counterfactual == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability_counterfactual <- invlogit(linear_predictor)
patient_data$outcomes_counterfactual <- ifelse(patient_data$outcome_probability_counterfactual >= 0.5, 'Positive', 'Negative')

crrt_patients <- patient_data %>%
  filter(treatment == 'CRRT')

categorical_vars <- c("male", "ethnicity", "race", "heart_comorb", 
                      "liver_comorb", "infection_comorb", "other_comorb", "prev_crrt")

model_data <- crrt_patients %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)

model_data1 <- patient_data %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)

noise_sd <- 0.5 
for(col in names(model_data)[sapply(model_data, is.numeric)]) {
  model_data[[col]] <- model_data[[col]] + rnorm(nrow(model_data), 0, noise_sd * sd(model_data[[col]]))
}

rf_model <- randomForest(
  outcome ~ .,
  data = model_data,
  ntree = 100,        
  mtry = 2,           
  maxnodes = 20,      
  importance = TRUE,
  proximity = TRUE
)

predictions <- predict(rf_model, model_data1, type = "prob")

patient_data$outcomes_predict <- predictions[,2]

plots_list <- list()
N_values <- seq(0.5, 0.9, by=0.05)

for(i in seq_along(N_values)) {
  N <- N_values[i]
  patient_data$Recommend <- ifelse(patient_data$outcomes_predict >=N, 'Positive', 'Negative')
  
  noncrrt_patients <- patient_data %>%
    filter(treatment == 'No_CRRT')
  
  heatmap_data1 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes_counterfactual)
  heatmap_data2 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes)
  
  heatmap_df1 <- as.data.frame(as.table(heatmap_data1))
  heatmap_df2 <- as.data.frame(as.table(heatmap_data2))
  
  names(heatmap_df1) <- c("Recommend", "CounterfactualOutcomes", "Count1")
  names(heatmap_df2) <- c("Recommend", "Outcomes", "Count2")
  
  combined_df <- heatmap_df1
  combined_df$Count2 <- heatmap_df2$Count2
  combined_df$label <- paste(combined_df$Count1,'/',combined_df$Count2)
  
  
  p <- ggplot(combined_df, aes(x = Recommend, y = CounterfactualOutcomes, fill = Count1)) +
    geom_tile(color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = label), color = "black") +
    labs(title = paste("Cutoff =", N),
         x = "Recommend",
         y = "Outcomes") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 10),
      plot.title = element_text(size = 12, face = "bold"),
      panel.grid = element_blank()
    )
  
  plots_list[[i]] <- p
}

combined_plot <- ggarrange(
  plotlist = plots_list,      
  ncol = 3,                   
  nrow = length(plots_list)/3, 
  labels = LETTERS[1:length(plots_list)], 
  common.legend = TRUE,       
  legend = "right"         
)
combined_plot <- annotate_figure(combined_plot,
                                 top = text_grob("Beneficial_CRRT_effect ", 
                                                 face = "bold", size = 14)
)
ggsave("Beneficial_CRRT_effect.jpg", combined_plot, width = 12, height = 10, dpi = 300)
pdf("Beneficial_CRRT_effect.pdf", width = 10, height = 8)
print(combined_plot)
dev.off()

#Detrimental CRRT effect

coefficients <- c(
  intercept = 8,
  treatmentCRRT = -1,      
  age = -0.002,
  maleMale = 0.02,
  ethnicityHispanic = 0.001,
  raceAsian = 0.01,
  raceBlack = 0.01,
  raceMultiple = 0.01,
  racePacific.Islander = 0.01,
  raceWhite =0.01,
  heart_comorbYes = -0.05,
  liver_comorbYes = -0.05,
  infection_comorbYes = -0.08,
  other_comorbYes = -0.02,
  prev_crrt = -0.03,
  creatinine = -0.08,
  bun = -0.008,
  map = 0.008,
  heart_rate = -0.08,
  tbil = -0.08,
  inr = -0.08,
  wbc = -0.008,
  pao2_fio2 = 0.008
)

invlogit <- function(x) {
  1 / (1 + exp(-x))
}

patient_data <- generate_covariates(10000) %>% clean_covariates()
patient_data$treatment <- rbinom(10000, 1, 0.5)
patient_data$treatment <- ifelse(patient_data$treatment == 1, 'CRRT', 'No_CRRT')
patient_data$treatment <- factor(patient_data$treatment, levels = c("No_CRRT", "CRRT"))
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                           coefficients["treatmentCRRT"] * (treatment == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability <- invlogit(linear_predictor)
patient_data$outcomes <- ifelse(patient_data$outcome_probability >= 0.5, 'Positive', 'Negative')

patient_data$treatment_counterfactual <- ifelse(patient_data$treatment == 'No_CRRT', "CRRT", 'No_CRRT')
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                           coefficients["treatmentCRRT"] * (treatment_counterfactual == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability_counterfactual <- invlogit(linear_predictor)
patient_data$outcomes_counterfactual <- ifelse(patient_data$outcome_probability_counterfactual >= 0.5, 'Positive', 'Negative')



crrt_patients <- patient_data %>%
  filter(treatment == 'CRRT')

categorical_vars <- c("male", "ethnicity", "race", "heart_comorb", 
                      "liver_comorb", "infection_comorb", "other_comorb", "prev_crrt")

model_data <- crrt_patients %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)

model_data1 <- patient_data %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)
noise_sd <- 0.5 
for(col in names(model_data)[sapply(model_data, is.numeric)]) {
  model_data[[col]] <- model_data[[col]] + rnorm(nrow(model_data), 0, noise_sd * sd(model_data[[col]]))
}

rf_model <- randomForest(
  outcome ~ .,
  data = model_data,
  ntree = 100,        
  mtry = 2,           
  maxnodes = 20,      
  importance = TRUE,
  proximity = TRUE
)

predictions <- predict(rf_model, model_data1, type = "prob")

patient_data$outcomes_predict <- predictions[,2]

plots_list <- list()
N_values <- seq(0.5, 0.9, by=0.05)

for(i in seq_along(N_values)) {
  N <- N_values[i]
  patient_data$Recommend <- ifelse(patient_data$outcomes_predict >=N, 'Positive', 'Negative')
  
  noncrrt_patients <- patient_data %>%
    filter(treatment == 'No_CRRT')
  
  heatmap_data1 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes_counterfactual)
  heatmap_data2 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes)
  
  heatmap_df1 <- as.data.frame(as.table(heatmap_data1))
  heatmap_df2 <- as.data.frame(as.table(heatmap_data2))
  
  names(heatmap_df1) <- c("Recommend", "CounterfactualOutcomes", "Count1")
  names(heatmap_df2) <- c("Recommend", "Outcomes", "Count2")
  
  combined_df <- heatmap_df1
  combined_df$Count2 <- heatmap_df2$Count2
  combined_df$label <- paste(combined_df$Count1,'/',combined_df$Count2)
  
  
  p <- ggplot(combined_df, aes(x = Recommend, y = CounterfactualOutcomes, fill = Count1)) +
    geom_tile(color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = label), color = "black") +
    labs(title = paste("Cutoff =", N),
         x = "Recommend",
         y = "Outcomes") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 10),
      plot.title = element_text(size = 12, face = "bold"),
      panel.grid = element_blank()
    )
  
  plots_list[[i]] <- p
}

combined_plot <- ggarrange(
  plotlist = plots_list,      
  ncol = 3,                   
  nrow = length(plots_list)/3, 
  labels = LETTERS[1:length(plots_list)], 
  common.legend = TRUE,       
  legend = "right"         
)
combined_plot <- annotate_figure(combined_plot,
                                 top = text_grob("Detrimental_CRRT_effect", 
                                                 face = "bold", size = 14)
)
ggsave("Detrimental_CRRT_effect.jpg", combined_plot, width = 12, height = 10, dpi = 300)
pdf("Detrimental_CRRT_effect.pdf", width = 10, height = 8)
print(combined_plot)
dev.off()

#crrt is Neutral
coefficients <- c(
  intercept = 7,
  treatmentCRRT = 0,      
  age = -0.002,
  maleMale = 0.02,
  ethnicityHispanic = 0.001,
  raceAsian = 0.01,
  raceBlack = 0.01,
  raceMultiple = 0.01,
  racePacific.Islander = 0.01,
  raceWhite =0.01,
  heart_comorbYes = -0.05,
  liver_comorbYes = -0.05,
  infection_comorbYes = -0.08,
  other_comorbYes = -0.02,
  prev_crrt = -0.03,
  creatinine = -0.08,
  bun = -0.008,
  map = 0.008,
  heart_rate = -0.08,
  tbil = -0.08,
  inr = -0.08,
  wbc = -0.008,
  pao2_fio2 = 0.008
)


patient_data <- generate_covariates(10000) %>% clean_covariates()
patient_data$treatment <- rbinom(10000, 1, 0.5)
patient_data$treatment <- ifelse(patient_data$treatment == 1, 'CRRT', 'No_CRRT')
patient_data$treatment <- factor(patient_data$treatment, levels = c("No_CRRT", "CRRT"))
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                           coefficients["treatmentCRRT"] * (treatment == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability <- invlogit(linear_predictor)
patient_data$outcomes <- ifelse(patient_data$outcome_probability >= 0.5, 'Positive', 'Negative')

patient_data$treatment_counterfactual <- ifelse(patient_data$treatment == 'No_CRRT', "CRRT", 'No_CRRT')
linear_predictor <- with(patient_data,
                         coefficients["intercept"] +
                           coefficients["treatmentCRRT"] * (treatment_counterfactual == "CRRT") +
                           coefficients["age"] * age +
                           coefficients["maleMale"] * (male == "Male") +
                           coefficients["ethnicityHispanic"] * (ethnicity == "Hispanic") +
                           coefficients["raceAsian"] * (race == "Asian") +
                           coefficients["raceBlack"] * (race == "Black") +
                           coefficients["raceMultiple"] * (race == "Multiple") +
                           coefficients["racePacific.Islander"] * (race == "Pacific Islander") +
                           coefficients["raceWhite"] * (race == "White") +
                           coefficients["heart_comorbYes"] * (heart_comorb == "Yes") +
                           coefficients["liver_comorbYes"] * (liver_comorb == "Yes") +
                           coefficients["infection_comorbYes"] * (infection_comorb == "Yes") +
                           coefficients["other_comorbYes"] * (other_comorb == "Yes") +
                           coefficients["prev_crrt"] * prev_crrt +
                           coefficients["creatinine"] * creatinine +
                           coefficients["bun"] * bun +
                           coefficients["map"] * map +
                           coefficients["heart_rate"] * heart_rate +
                           coefficients["tbil"] * tbil +
                           coefficients["inr"] * inr +
                           coefficients["wbc"] * wbc +
                           coefficients["pao2_fio2"] * pao2_fio2
)

patient_data$outcome_probability_counterfactual <- invlogit(linear_predictor)
patient_data$outcomes_counterfactual <- ifelse(patient_data$outcome_probability_counterfactual >= 0.5, 'Positive', 'Negative')

crrt_patients <- patient_data %>%
  filter(treatment == 'CRRT')

categorical_vars <- c("male", "ethnicity", "race", "heart_comorb", 
                      "liver_comorb", "infection_comorb", "other_comorb", "prev_crrt")

model_data <- crrt_patients %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)

model_data1 <- patient_data %>%
  mutate(across(all_of(categorical_vars), as.factor),
         outcome = factor(outcomes)) %>%
  select(age, male, ethnicity, race, heart_comorb, liver_comorb,
         infection_comorb, other_comorb, prev_crrt, creatinine,
         bun, map, heart_rate, tbil, inr, wbc, pao2_fio2, outcome)
noise_sd <- 0.5 
for(col in names(model_data)[sapply(model_data, is.numeric)]) {
  model_data[[col]] <- model_data[[col]] + rnorm(nrow(model_data), 0, noise_sd * sd(model_data[[col]]))
}

rf_model <- randomForest(
  outcome ~ .,
  data = model_data,
  ntree = 100,        
  mtry = 2,           
  maxnodes = 20,      
  importance = TRUE,
  proximity = TRUE
)

predictions <- predict(rf_model, model_data1, type = "prob")

patient_data$outcomes_predict <- predictions[,2]

plots_list <- list()
N_values <- seq(0.5, 0.9, by=0.05)

for(i in seq_along(N_values)) {
  N <- N_values[i]
  patient_data$Recommend <- ifelse(patient_data$outcomes_predict >=N, 'Positive', 'Negative')
  
  noncrrt_patients <- patient_data %>%
    filter(treatment == 'No_CRRT')
  
  heatmap_data1 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes_counterfactual)
  heatmap_data2 <- table(noncrrt_patients$Recommend, noncrrt_patients$outcomes)
  
  heatmap_df1 <- as.data.frame(as.table(heatmap_data1))
  heatmap_df2 <- as.data.frame(as.table(heatmap_data2))
  
  names(heatmap_df1) <- c("Recommend", "CounterfactualOutcomes", "Count1")
  names(heatmap_df2) <- c("Recommend", "Outcomes", "Count2")
  
  combined_df <- heatmap_df1
  combined_df$Count2 <- heatmap_df2$Count2
  combined_df$label <- paste(combined_df$Count1,'/',combined_df$Count2)
  
  
  p <- ggplot(combined_df, aes(x = Recommend, y = CounterfactualOutcomes, fill = Count1)) +
    geom_tile(color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = label), color = "black") +
    labs(title = paste("Cutoff =", N),
         x = "Recommend",
         y = "Outcomes") +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 10),
      plot.title = element_text(size = 12, face = "bold"),
      panel.grid = element_blank()
    )
  
  plots_list[[i]] <- p
}

combined_plot <- ggarrange(
  plotlist = plots_list,      
  ncol = 3,                   
  nrow = length(plots_list)/3, 
  labels = LETTERS[1:length(plots_list)], 
  common.legend = TRUE,       
  legend = "right"         
)
combined_plot <- annotate_figure(combined_plot,
                                 top = text_grob("Neutral_CRRT_effect", 
                                                 face = "bold", size = 14)
)
ggsave("Neutral_CRRT_effect.jpg", combined_plot, width = 12, height = 10, dpi = 300)
pdf("Neutral_CRRT_effect.pdf", width = 10, height = 8)
print(combined_plot)
dev.off()