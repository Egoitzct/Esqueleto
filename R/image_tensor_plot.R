#' @title Plot image tensor
#' @import torch
#' @import torchvision

image_tensor_plot <- function(image_path) {
  if (torch::torch_is_installed() == FALSE) {
    torch::install_torch()
  }

  train_ds <- image_loading(image_path)$train_ds

  image_dataloader <- dataloader(train_ds, batch_size = 64, shuffle = TRUE)

  batch <- image_dataloader$.iter()$.next()

  class_names <- train_ds$classes

  classes <- batch[[2]]
  classes

  images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
  mean <- c(0.485, 0.456, 0.406)
  std <- c(0.229, 0.224, 0.225)
  images <- std * images + mean
  images <- images * 255
  images[images > 255] <- 255
  images[images < 0] <- 0

  par(mfcol = c(4, 6), mar = rep(1, 4))

  images %>%
    purrr::array_tree(1) %>%
    purrr::set_names(class_names[as_array(classes)]) %>%
    purrr::map(as.raster, max = 255) %>%
    purrr::iwalk(~{plot(.x); title(.y)})
}
