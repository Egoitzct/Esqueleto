#' @title image_train_learning_rate
#' @author Egoitz Carral
#' @import luz
#' @import torch
#'
#' @export
#'

image_train_lr <- function(image_path, model = "resnet34", pretrained = FALSE, batch_size = 128) {

  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  image_data <- image_loading(image_path)

  train_ds <- image_data$train_ds
  train_dl <- dataloader(train_ds, batch_size = batch_size)

  if (model == "alexnet") {
    net <- alexnet()

    num_classes <- length(train_ds$classes)

    if (num_classes == 2) {
      model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = num_classes)

      learning_rate <- lr_finder(
        object = model,
        data = train_ds,
        verbose = TRUE,
        dataloader_options = list(batch_size = batch_size),
        start_lr = 1e-6,
        end_lr = 1
      )
      plot(learning_rate)+
        ggplot2::coord_cartesian(ylim = c(NA, 5))

      str(learning_rate)

    } else {
      model <- net %>%
        setup(
          loss = nn_cross_entropy_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes)

      learning_rate <- lr_finder(
        object = model,
        data = train_ds,
        verbose = TRUE,
        dataloader_options = list(batch_size = batch_size),
        start_lr = 1e-6,
        end_lr = 1
      )
      plot(learning_rate) +
        ggplot2::coord_cartesian(ylim = c(NA, 5))

      str(learning_rate)
    }

  } else if (model == "resnet34") {

    num_classes <- length(train_ds$classes)

    if (num_classes == 2) {

      net <- torch::nn_module(
        initialize = function(num_classes) {
          self$model <- model_resnet34(pretrained = pretrained)

          for (par in self$parameters) {
            par$requires_grad_(FALSE)
          }
        },
        forward = function(x) {
          self$model(x)[,1]
        }
      )

      model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = num_classes)

      learning_rate <- lr_finder(
        object = model,
        data = train_dl,
        verbose = TRUE,
        start_lr = 1e-6,
        end_lr = 1
      )
      plot(learning_rate)+
        ggplot2::coord_cartesian(ylim = c(NA, 5))

      str(learning_rate)

    } else {

      net <- torch::nn_module(
        initialize = function(num_classes) {
          self$model <- model_resnet34(pretrained = pretrained)

          for (par in self$parameters) {
            par$requires_grad_(FALSE)
          }

          num_features <- self$model$fc$in_features

          self$model$fc <- nn_linear(in_features = num_features, out_features = num_classes)

        },
        forward = function(x) {
          self$model(x)[,1]
        }
      )

      model <- net %>%
        setup(
          loss = nn_cross_entropy_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes)

      learning_rate <- lr_finder(
        object = model,
        data = train_dl,
        verbose = TRUE,
        start_lr = 1e-6,
        end_lr = 1
      )
      plot(learning_rate) +
        ggplot2::coord_cartesian(ylim = c(NA, 5))

      str(learning_rate)
    }

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
}
