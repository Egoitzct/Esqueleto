#' @title Entrenamiento de imágenes
#' @import torch
#' @import torchvision
#' @import luz
#'
#' @export

image_train <- function(image_path, model = "resnet34", pretrained = FALSE, batch_size = 128) {
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
    num_classes <- length(train_ds$classes)

    net <- alexnet()

    if (num_classes == 2) {
      model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE) #Cambiar epochs después de pruebas
    } else {
      model <- net %>%
        setup(
          loss = nn_cross_entropy_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE)
    }

  } else if (model == "resnet34") {
    net <- torchvision::model_resnet34(pretrained = pretrained)
  } else if (model == "resnet50") {
    net <- torchvision::model_resnet50(pretrained = pretrained)
  } else if (model == "resnet101") {
    net <- torchvision::model_resnet101(pretrained = pretrained)
  } else if (model == "resnet152") {
    net <- torchvision::model_resnet152(pretrained = pretrained)
  } else if (model == "resnext50_32x4d") {
    net <- torchvision::model_resnext50_32x4d(pretrained = pretrained)
  } else if (model == "resnext101_32x8d") {
    net <- torchvision::model_resnext101_32x8d(pretrained = pretrained)
  } else if (model == "wide_resnet50_2") {
    net <- torchvision::model_wide_resnet50_2(pretrained = pretrained)
  } else if (model == "wide_resnet101_2") {
    net <- torchvision::model_wide_resnet101_2(pretrained = pretrained)
  }

  return(model)

}
