#' @title Entrenamiento de imágenes
#' @import torch
#' @import torchvision
#' @import luz
#'

image_train <- function(image_path, model = "resnet34", pretrained = FALSE, batch_size = 64) {
  if (torch::torch_is_installed() == FALSE) {
    torch::install_torch()
  }

  device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  image_data <- image_loading(image_path)

  train_ds <- image_data$train_ds
  valid_ds <- image_data$valid_ds
  test_ds <- image_data$test_ds

  train_dl <- dataloader(train_ds, batch_size = batch_size)
  valid_dl <- dataloader(valid_ds, batch_size = batch_size)
  test_dl <- dataloader(test_ds, batch_size = batch_size)

  if (model == "alexnet") {
    net <- alexnet()
  } else if (model == "resnet34") {
    net <- resnet34()
  } else if (model == "resnet50") {
    net <- resnet50()
  } else if (model == "resnet101") {
    net <- resnet101()
  } else if (model == "resnet152") {
    net <- resnet152()
  } else if (model == "resnext50_32x4d") {
    net <- resnext50_32x4d()
  } else if (model == "resnext101_32x8d") {
    net <- resnext101_32x8d()
  } else if (model == "wide_resnet50_2") {
    net <- wide_resnet50_2()
  } else if (model == "wide_resnet101_2") {
    net <- wide_resnet101_2()
  }

  num_classes <- length(train_ds$classes)

  model <- net %>%
    setup(
      loss = nn_cross_entropy_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_binary_accuracy_with_logits()
      )
    ) %>%
    set_hparams(num_classes) %>%
    luz::fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE)

  path <- paste(getwd(), "model", sep = "/")

  luz_save(model, path = path)

  }
