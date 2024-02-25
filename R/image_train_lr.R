#' @title image_train_learning_rate
#' @author Egoitz Carral
#' @import luz
#' @import torch
#'
#' @export
#'

image_train_lr <- function(image_path, model = "alexnet", pretrained = FALSE, batch_size = 128) {

  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  image_data <- image_loading(image_path)

  train_ds <- image_data$train_ds
  train_dl <- dataloader(train_ds, batch_size = batch_size)

  if (model == "alexnet") {
    net <- torch::nn_module(
      "AlexNet",
      initialize = function(num_classes) {
        self$model <- alexnet_mod(pretrained = pretrained)

        for (par in self$parameters) {
          par$requires_grad_(FALSE)
        }

        self$model$classifier <- nn_sequential(
          nn_dropout(0.5),
          nn_linear(9216, 512),
          nn_relu(),
          nn_linear(512, 256),
          nn_relu(),
          nn_linear(256, num_classes)
        )
      },
      forward = function(x) {
        self$model(x)
      }
    )

    num_classes <- length(train_ds$classes)

    if (num_classes == 0) {
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
    }
  }
}
