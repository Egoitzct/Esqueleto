#' @title Entrenamiento de imágenes
#' @import torch
#' @import torchvision
#' @import luz
#'
#' @export

image_train <- function(image_path, model = "alexnet", pretrained = FALSE, batch_size = 64, epochs = 10) {
  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  image_data <- image_loading(image_path)

  train_ds <- image_data$train_ds
  valid_ds <- image_data$valid_ds
  test_ds <- image_data$test_ds

  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)
  test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)

  num_classes <- length(train_ds$classes)

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

    if (num_classes == 0) {
      fitted_model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE) #Cambiar epochs después de pruebas
    } else {
      fitted_model <- net %>%
        setup(
          loss = nn_cross_entropy_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = 0.01) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE)
    }

  }

  model_name <- paste("trained", model, sep = "_")

  model_name <- paste(model_name, as.character(Sys.Date()), sep = "_")

  model_name <- paste(model_name, "pt", sep = ".")

  luz::luz_save(fitted_model, paste(as.character(getwd()), model_name, sep = "/"))

}
