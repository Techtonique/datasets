all_datasets <- readRDS(file = "all_datasets.rds")

names_all_datasets <- names(all_datasets)

ts_characteristics <- matrix(NA, nrow = length(all_datasets), 
                          ncol = 17)
rownames(ts_characteristics) <- names_all_datasets

# Load necessary libraries
#install.packages("fBasics")   # Only needed if not already installed
#install.packages("tseries")
#library(fBasics)
#library(tseries)
library(urca)

# Define function
time_series_summary <- function(ts_data) {
  # Check if input is a time series object
  if (!is.ts(ts_data)) {
    return("Input data is not a time series object")
  }
  
  # Basic summary statistics
  additional_stats <- basicStats(ts_data)
  
  
  test <- Box.test(ts_data)$p.value
  
  names_additional_stats <- rownames(additional_stats)
  results <- round(c(unlist(additional_stats), test), 2)
  names(results) <- c(names_additional_stats, "Box-test p-value")
  return(results)
}

for (i in seq_len(length(all_datasets)))
{
  ts_characteristics[i, ] <- time_series_summary(all_datasets[[i]])
}

colnames(ts_characteristics) <- names(time_series_summary(all_datasets[[1]]))

print(ts_characteristics)