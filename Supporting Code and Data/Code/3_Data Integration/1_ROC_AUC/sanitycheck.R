title: "Lasso - Liver"
author: "Tirdod Behbehani, Elisa Scocco,Iñigo Exposito"
date: "2025-04-19"
output: html_document
---
  
  

#––– PACKAGES –––
library(readr)
library(glmnet)
library(pROC)
library(caret)
library(ParBayesianOptimization)
library(dplyr)




#––– 0) LOAD + PREP DATA –––
gene_expression_stage_34 <- read_csv("/Users/usuario/OneDrive/Desktop/Thesis/personal_files/Liver_final/gene_expression_only_liver_final.csv") %>%
  select(-Is_stage_34, -id)

# Outcome and predictor matrix
y <- gene_expression_stage_34$OS
X <- gene_expression_stage_34 %>% select(-OS)

# Standardize features
X <- scale(X) %>% as.matrix()
predictor_names <- colnames(X)

# Load the coefficients from Adaptive Ridge and remove first uninformative 25%
coef_enet <- read_csv("/Users/usuario/OneDrive/Desktop/Thesis/personal_files/breastselected/OS_coefficients_adaptive_Ridge.csv")

# Rank by absolute value of coefficients
coef_enet <- coef_enet %>%
  mutate(abs_value = abs(value)) %>%
  arrange(desc(abs_value)) %>%
  # Keep only the top 75% of genes
  slice(1:floor(0.75 * n())) %>%
  mutate(selected = 1) %>%
  select(Variable, selected)

# Build z_j: binary vector of selected features
p <- ncol(X)
z_j <- integer(p); names(z_j) <- predictor_names
z_j[ intersect(predictor_names, coef_enet$Variable) ] <-
  coef_enet$selected[ match(intersect(predictor_names, coef_enet$Variable),
                            coef_enet$Variable) ]




set.seed(236)
n <- nrow(X)
train_idx <- sample(1:n, size = floor(0.8 * n))
X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]

z_j_train <- z_j  # same features for both train/test



#––– 1) Pre‐define CV folds once –––
set.seed(123)
k     <- 10
folds <- sample(rep(1:k, length.out = nrow(X_train)))





#––– 2) Objective: 3‐step integrative LASSO + CV‐AUC –––
lasso_cv_loglik_train <- function(theta0) {
  lambda_scalar <- exp(theta0)
  
  logliks <- numeric(k)
  for (i in seq_len(k)) {
    tr <- which(folds != i); te <- which(folds == i)
    X_tr <- X_train[tr, , drop = FALSE]; y_tr <- y_train[tr]
    X_te <- X_train[te, , drop = FALSE]; y_te <- y_train[te]
    
    fit <- glmnet(x = X_tr, y = y_tr,
                  family = "binomial",
                  alpha = 1, lambda = lambda_scalar,
                  intercept = TRUE,
                  standardize = FALSE)
    
    coefs <- as.numeric(coef(fit, s = lambda_scalar))
    intercept <- coefs[1]
    beta_tilde <- coefs[-1]
    
    linpred <- intercept + X_te %*% beta_tilde
    prob_te <- 1 / (1 + exp(-linpred))
    eps <- 1e-8
    loglik <- y_te * log(pmax(prob_te, eps)) + (1 - y_te) * log(pmax(1 - prob_te, eps))
    logliks[i] <- mean(loglik)
  }
  
  mean(logliks)
}





#we include the vlaue of theta0_vals for which we obtained the highest AUC in LASSO model for only Liver cancer

#theta0_vals <- sort(unique(c(seq(-3, 3, by = 0.25), log(0.0261021))))  
#theta1_vals <- seq(-1.5, 1.5, by = 0.25)

theta0_vals <- seq(-5, 0, by = 0.25)
cv_results <- tibble(theta0 = theta0_vals) %>%
  rowwise() %>%
  mutate(mean_loglik = lasso_cv_loglik_train(theta0)) %>%
  ungroup()

best_result <- cv_results %>% filter(mean_loglik == max(mean_loglik))
lambda_custom <- exp(best_result$theta0)


cv_lasso <- cv.glmnet(X_train, y_train, family = "binomial",
                      alpha = 1, foldid = folds, type.measure = "deviance")

lambda_cvglmnet <- cv_lasso$lambda.min

beta_custom <- as.numeric(coef(glmnet(X_train, y_train, family = "binomial",
                                      lambda = lambda_custom, standardize = FALSE)))
beta_cvglmnet <- as.numeric(coef(cv_lasso, s = "lambda.min"))

cat("Sum of absolute differences in coefficients:", sum(abs(beta_custom - beta_cvglmnet)), "\n")
 

cat("Best lambda from custom:", round(lambda_custom, 6), "\n")
cat("Lambda from cv.glmnet:", round(lambda_cvglmnet, 6), "\n")
cat("Difference:", abs(lambda_custom - lambda_cvglmnet), "\n")



#I validated my lasso_cv_loglik_train implementation against cv.glmnet using the same folds and log-likelihood. The selected lambdas and resulting coefficients are nearly identical (λ difference ≈ 0.0041, coefficient L1 diff ≈ 0.039).


