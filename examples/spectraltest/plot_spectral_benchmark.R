#!/usr/bin/env Rscript
# Figures for spectral_benchmark.py: standard vs spectral BEAGLE CPU implementations. Base R only.
#
# usage: Rscript plot_spectral_benchmark.R bench.csv [more.csv ...] outdir
#
# Several CSVs are combined (e.g. a full run and a likelihood-only extension); a configuration in more than one
# keeps the row with gradient times, and rows without them count as missing for the gradient.
# For each model / tip data / tree size / vector group of the CSV written by `spectral_benchmark.py run`:
#   speedup_<group>.pdf/png    heat maps of the speedup (standard time / spectral time) over states x patterns,
#                              for the likelihood and the gradient, K = 1 and 4; blue: spectral faster
#   crossover_<group>.pdf/png  the pattern count at which the standard implementation becomes faster, per state
#                              count (open triangles: spectral faster over the whole range)
#   times_<group>.pdf/png      time per evaluation against patterns for a few state counts

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript plot_spectral_benchmark.R bench.csv [more.csv ...] outdir")
d <- do.call(rbind, lapply(args[-length(args)], read.csv, stringsAsFactors = FALSE))
out <- args[length(args)]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

gradient <- d$pre_ms > 0 | d$adjoint_ms > 0
d <- d[order(gradient), ] # rows with gradient times last, so they are kept
key <- paste(d$impl, d$vector, d$states, d$patterns, d$categories, d$tips, d$model, d$tipdata)
d <- d[!duplicated(key, fromLast = TRUE), ]
nogradient <- !(d$pre_ms > 0 | d$adjoint_ms > 0)
for (column in c("gradient_ms", "pre_ms", "adjoint_ms")) d[nogradient, column] <- NA

# pattern count at which the speedup first falls below 1 (log-linear interpolation); NA with a censoring flag
crossover <- function(patterns, speedup) {
  o <- order(patterns); p <- patterns[o]; s <- speedup[o]
  keep <- is.finite(s) & s > 0; p <- p[keep]; s <- s[keep]
  if (length(p) == 0) return(c(value = NA, censored = NA))
  if (s[1] < 1) return(c(value = p[1], censored = -1))
  for (i in seq_len(length(p) - 1)) {
    if (s[i + 1] < 1) {
      f <- log(s[i]) / (log(s[i]) - log(s[i + 1]))
      return(c(value = exp(log(p[i]) + f * (log(p[i + 1]) - log(p[i]))), censored = 0))
    }
  }
  c(value = p[length(p)], censored = 1)
}

both <- function(f) {
  for (device in c("pdf", "png")) {
    if (device == "pdf") pdf(paste0(f$name, ".pdf"), width = f$width, height = f$height)
    else png(paste0(f$name, ".png"), width = f$width * 110, height = f$height * 110, res = 110)
    f$draw()
    dev.off()
  }
}

palette <- colorRampPalette(c("#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"))(101)

groups <- unique(d[, c("model", "tipdata", "tips", "vector")])
for (gi in seq_len(nrow(groups))) {
  gr <- groups[gi, ]
  g <- d[d$model == gr$model & d$tipdata == gr$tipdata & d$tips == gr$tips & d$vector == gr$vector, ]
  tag <- paste(gr$model, gr$tipdata, paste0(gr$tips, "tips"), gr$vector, sep = "_")
  title <- sprintf("%s model, tips as %s, %d tips, %s", gr$model, gr$tipdata, gr$tips, gr$vector)
  std <- g[g$impl == "standard", ]
  spe <- g[g$impl == "spectral", ]
  m <- merge(std, spe, by = c("states", "patterns", "categories"), suffixes = c(".std", ".spe"))
  if (nrow(m) == 0) next
  for (metric in c("likelihood_ms", "gradient_ms", "post_ms", "pre_ms", "adjoint_ms")) {
    m[[paste0("speedup.", metric)]] <- m[[paste0(metric, ".std")]] / m[[paste0(metric, ".spe")]]
  }
  states <- sort(unique(m$states))
  patterns <- sort(unique(m$patterns))
  categories <- sort(unique(m$categories))

  # heat maps
  both(list(name = file.path(out, paste0("speedup_", tag)), width = 5.2 * length(categories), height = 9.5,
    draw = function() {
      par(mfrow = c(2, length(categories)), mar = c(4.2, 4.2, 3, 1), oma = c(0, 0, 2, 0))
      for (metric in c("likelihood_ms", "gradient_ms")) for (k in categories) {
        z <- matrix(NA, length(patterns), length(states))
        for (i in seq_along(patterns)) for (j in seq_along(states)) {
          r <- m[m$patterns == patterns[i] & m$states == states[j] & m$categories == k, ]
          if (nrow(r) == 1) z[i, j] <- log2(r[[paste0("speedup.", metric)]])
        }
        lim <- 6
        image(seq_along(patterns), seq_along(states), pmax(pmin(z, lim), -lim), col = palette,
              zlim = c(-lim, lim), axes = FALSE, xlab = "patterns", ylab = "states",
              main = sprintf("%s, K = %d", sub("_ms", "", metric), k))
        axis(1, at = seq_along(patterns), labels = patterns, las = 2, cex.axis = 0.8)
        axis(2, at = seq_along(states), labels = states, las = 1, cex.axis = 0.8)
        box()
        if (any(is.finite(z))) contour(seq_along(patterns), seq_along(states), z, levels = 0, add = TRUE,
                                       drawlabels = FALSE, lwd = 2)
        for (i in seq_along(patterns)) for (j in seq_along(states)) if (is.finite(z[i, j])) {
          v <- 2^z[i, j]
          text(i, j, if (v >= 10) sprintf("%.0f", v) else sprintf("%.1f", v), cex = 0.62)
        }
      }
      mtext(paste("Speedup of spectral over standard (blue: spectral faster; line: equal) -", title),
            outer = TRUE, cex = 0.9)
    }))

  # crossovers
  both(list(name = file.path(out, paste0("crossover_", tag)), width = 7, height = 5.5, draw = function() {
    par(mar = c(4.2, 4.5, 3, 1))
    plot(NULL, xlim = range(states), ylim = range(patterns), log = "xy", xlab = "states",
         ylab = "pattern count at the crossover", main = paste("Standard becomes faster above the line -", title),
         cex.main = 0.85)
    grid()
    styles <- expand.grid(metric = c("likelihood_ms", "gradient_ms"), k = categories, stringsAsFactors = FALSE)
    colors <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
    for (si in seq_len(nrow(styles))) {
      metric <- styles$metric[si]; k <- styles$k[si]
      cs <- t(sapply(states, function(s) {
        r <- m[m$states == s & m$categories == k, ]
        crossover(r$patterns, r[[paste0("speedup.", metric)]])
      }))
      ok <- cs[, "censored"] == 0 & is.finite(cs[, "value"])
      lines(states[ok], cs[ok, "value"], col = colors[si], lwd = 2, lty = if (metric == "gradient_ms") 1 else 2)
      points(states[ok], cs[ok, "value"], col = colors[si], pch = 19, cex = 0.7)
      up <- which(cs[, "censored"] == 1)
      if (length(up)) points(states[up], cs[up, "value"], col = colors[si], pch = 2)
    }
    legend("topleft", bty = "n", cex = 0.8, col = colors[seq_len(nrow(styles))], lwd = 2,
           lty = ifelse(styles$metric == "gradient_ms", 1, 2),
           legend = sprintf("%s, K = %d", sub("_ms", "", styles$metric), styles$k))
  }))

  # times against patterns
  pick <- states[states %in% c(4, 8, 20, 32, 61, 128)]
  if (length(pick) == 0) pick <- states[unique(round(seq(1, length(states), length.out = 4)))]
  both(list(name = file.path(out, paste0("times_", tag)), width = 5 * length(categories), height = 9.5,
    draw = function() {
      par(mfrow = c(2, length(categories)), mar = c(4.2, 4.5, 3, 1), oma = c(0, 0, 2, 0))
      colors <- setNames(hcl.colors(length(pick), "Dark 3"), pick)
      for (metric in c("likelihood_ms", "gradient_ms")) for (k in categories) {
        sub <- g[g$categories == k & g$states %in% pick, ]
        if (all(is.na(sub[[metric]]))) next
        plot(NULL, xlim = range(sub$patterns), ylim = range(sub[[metric]], na.rm = TRUE), log = "xy",
             xlab = "patterns",
             ylab = "ms per evaluation", main = sprintf("%s, K = %d", sub("_ms", "", metric), k))
        grid()
        for (s in pick) for (impl in c("standard", "spectral")) {
          r <- sub[sub$states == s & sub$impl == impl, ]
          r <- r[order(r$patterns), ]
          lines(r$patterns, r[[metric]], col = colors[as.character(s)], lwd = 2,
                lty = if (impl == "standard") 1 else 2)
        }
        legend("topleft", bty = "n", cex = 0.75, legend = c(paste("S =", pick), "standard", "spectral"),
               col = c(colors, "black", "black"), lwd = 2, lty = c(rep(1, length(pick)), 1, 2))
      }
      mtext(paste("Time per evaluation -", title), outer = TRUE, cex = 0.9)
    }))
}
cat("figures written to", out, "\n")
