# Base risk score with and without the HADS depression factor (predictpd R package).
# Usage: Rscript riskscore_nodep_compute.R <inputs.csv> <output.csv> <predictpd R dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: riskscore_nodep_compute.R <inputs.csv> <output.csv> <R dir>")
in_csv <- args[1]
out_csv <- args[2]
lib_dir <- args[3]
source(file.path(lib_dir, "predictpd_risk.r"))
source(file.path(lib_dir, "predictpd_risk_from_data.r"))

dat <- read.csv(in_csv, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows, %d cols\n", nrow(dat), ncol(dat)))

sessions <- c("4", "5", "7", "8")

for (s in sessions) {
  common <- list(
    data               = dat,
    age_col            = paste0("age_session_", s),
    gender_col         = "gender",
    smoking_col        = paste0("smoking_status_", s),
    coffee_col         = paste0("drink_coffee_", s),
    fx_firstdegree_col = "fx_firstdegree",
    constipation_col   = paste0("use_laxatives_more_than_1x_per_week_", s),
    ed_col             = paste0("maintain_erection_untreated_poor_", s),
    hads_dep_col       = paste0("hads_depression_level_moderatesevere_", s),
    rbd_col            = paste0("rbdsq_summary_score_highrisk_", s),
    headinjury_col     = paste0("headinjury_loss_consiousness_", s),  # dataset spelling
    nsaids_col         = paste0("rx_nsaids_use_", s),
    ca_blocker_col     = paste0("rx_calcium_channel_bocker_use_", s),
    beta_blocker_col   = paste0("rx_beta_blocker_use_", s),
    alcohol_col        = paste0("drink_alcohol_", s),
    diabetes_col       = "hx_diabetes",
    diabetes_age_col   = "hx_diabetes_age",
    pesticide_col      = "pesticide_exposure_ever"
  )

  dat <- do.call(predictpd_risk_from_data,
                 c(common, list(output_col = paste0("lib_base_", s))))
  common$data <- dat
  dat <- do.call(predictpd_risk_from_data,
                 c(common, list(output_col = paste0("lib_nodep_", s),
                                include_hads = FALSE)))
  cat(sprintf("Session %s: base non-NA %d, nodep non-NA %d\n",
              s, sum(!is.na(dat[[paste0("lib_base_", s)]])),
              sum(!is.na(dat[[paste0("lib_nodep_", s)]]))))
}

out_cols <- c("HashFinal",
              paste0("lib_base_", sessions),
              paste0("lib_nodep_", sessions))
write.csv(dat[, out_cols], out_csv, row.names = FALSE)
cat("Written", out_csv, "\n")
