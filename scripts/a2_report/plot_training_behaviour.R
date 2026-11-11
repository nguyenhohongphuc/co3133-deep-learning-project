# Regenerate the report figures from the repository root:
# Rscript scripts/a2_report/plot_training_behaviour.R
# Requires jsonlite. Figures use R's PDF graphics device.

protocol <- jsonlite::fromJSON("results/a2/tables/a2_test_protocol.json")
output_dir <- "reports/a2/assets/training"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

seed_colors <- c("42" = "#216582", "43" = "#AD4C20", "44" = "#73539C")
runs <- lapply(protocol$run_ids, function(run_id) {
    directory <- file.path("results/a2/runs", run_id)
    history <- read.csv(file.path(directory, "history.csv"))
    metadata <- jsonlite::fromJSON(file.path(directory, "metrics.json"))
    selected <- metadata$best_epoch + 1L
    stopifnot(
        identical(history$epoch, seq_len(nrow(history)) - 1L),
        nrow(history) == metadata$epochs_run,
        selected == which.max(history$val_macro_f1)
    )
    history$epoch <- history$epoch + 1L
    list(history = history, model = metadata$model, selected = selected,
         color = seed_colors[[as.character(metadata$seed)]])
})

draw_panel <- function(runs, title, ticks, limits, loss) {
    label <- if (loss) "Cross-entropy" else "Validation macro-F1"
    plot(NA, xlim = c(1, max(ticks)), ylim = limits, axes = FALSE,
         xlab = "Epoch", ylab = label)
    y_ticks <- pretty(limits, n = 5)
    abline(h = y_ticks, col = "#DFE6EB", lwd = 0.7)
    axis(1, at = ticks, col = "#A4AFB8")
    axis(2, at = y_ticks, las = 1, col = "#A4AFB8")
    box(bty = "l", col = "#A4AFB8")
    title(main = title, adj = 0, line = 0.8, cex.main = 1)

    for (run in runs) {
        h <- run$history
        if (loss) {
            lines(h$epoch, h$train_loss, col = run$color, lty = 2, lwd = 1.65)
            lines(h$epoch, h$val_loss, col = run$color, lwd = 1.85)
        } else {
            lines(h$epoch, h$val_macro_f1, col = run$color, lwd = 1.85)
            points(run$selected, h$val_macro_f1[run$selected], pch = 21,
                   col = run$color, bg = "white", lwd = 1.45, cex = 0.94)
            points(tail(h$epoch, 1), tail(h$val_macro_f1, 1), pch = 4,
                   col = run$color, lwd = 1.45, cex = 0.94)
        }
    }
}

draw_row <- function(runs, name, ticks, loss_limits, f1_limits) {
    par(mar = c(3.5, 4.5, 2.0, 0.8), mgp = c(2.7, 0.55, 0), tcl = -0.22)
    draw_panel(runs, paste(name, "loss"), ticks, loss_limits, loss = TRUE)
    draw_panel(runs, paste(name, "validation score"), ticks, f1_limits, loss = FALSE)
}

draw_key <- function(show_seeds) {
    par(mar = rep(0, 4))
    plot.new()
    if (show_seeds) {
        legend("top", legend = paste("Seed", names(seed_colors)), col = seed_colors,
               lty = 1, lwd = 2, horiz = TRUE, bty = "n", cex = 0.93)
    }
    legend(if (show_seeds) "bottom" else "center",
           legend = c("Training loss", "Validation loss", "Selected checkpoint", "Last epoch"),
           lty = c(2, 1, NA, NA), pch = c(NA, NA, 21, 4), pt.bg = "white",
           col = "#253649", lwd = 1.65, horiz = TRUE, bty = "n", cex = 0.93)
}

cnn <- Filter(function(run) run$model == "cnn", runs)
frozen <- Filter(function(run) run$model == "resnet50_frozen", runs)
full <- Filter(function(run) run$model == "resnet50_full", runs)
stopifnot(length(cnn) == 1, length(frozen) == 3, length(full) == 3)

pdf(file.path(output_dir, "a2_cnn_training.pdf"), width = 8.8, height = 3.0,
    pointsize = 12, useDingbats = FALSE)
layout(rbind(c(1, 2), c(3, 3)), heights = c(5.7, 0.85))
draw_row(cnn, "CNN", c(1, 4, 8, 12, 16, 20), c(1.5, 3.08), c(0.15, 0.59))
draw_key(show_seeds = FALSE)
invisible(dev.off())

pdf(file.path(output_dir, "a2_resnet_training.pdf"), width = 8.8, height = 5.7,
    pointsize = 12, useDingbats = FALSE)
layout(rbind(c(1, 2), c(3, 4), c(5, 5)), heights = c(5, 5, 1.35))
ticks <- c(1, 4, 8, 12, 15)
draw_row(frozen, "Frozen ResNet-50", ticks, c(0.2, 2), c(0.64, 0.90))
draw_row(full, "Full ResNet-50", ticks, c(0.2, 2), c(0.64, 0.90))
draw_key(show_seeds = TRUE)
invisible(dev.off())
