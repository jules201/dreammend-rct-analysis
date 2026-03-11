# Sample size calculation for a two-sample t-test

# Load power analysis package

library(pwr)

# Define parameters
effect_size <- 2 / 2  # Delta / SD = 1.0
power <- 0.90
alpha <- 0.05
dropout_rate <- 0.20  # 20% expected dropout

# Compute required sample size per group
sample <- pwr.t.test(d = effect_size, 
                     power = power, 
                     sig.level = alpha, 
                     type = "two.sample", 
                     alternative = "two.sided")

# Print result
print(sample)

# Adjust for dropout
adjusted_n <- ceiling(sample$n / (1 - dropout_rate))
cat("Adjusted sample size per group (20% dropout):", adjusted_n, "\n")
cat("Total sample size needed:", adjusted_n * 2, "\n")
