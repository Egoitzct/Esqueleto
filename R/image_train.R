#' @title Entrenamiento de imágenes
#' @import torch
#' @import torchvision
#' @import luz
#'
#' @export

image_train <- function(image_path, model = "alexnet", pretrained = FALSE, batch_size = 64) {
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
        self$model(x)[,1]
      }
    )

    if (num_classes == 0) {
      model <- net %>%
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
      model <- net %>%
        setup(
          loss = nn_nll_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE)
    }

  } else if (model == "vgg11") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- model_vgg11(pretrained = FALSE)

        for (par in self$parameters) {
          par$requires_grad_(FALSE)
        }

        self$avgpool <- torch::nn_adaptive_avg_pool2d(c(7,7))
        self$classifier <- torch::nn_sequential(
          torch::nn_linear(512 * 7 * 7, 4096),
          torch::nn_relu(TRUE),
          torch::nn_dropout(),
          torch::nn_linear(4096, 4096),
          torch::nn_relu(TRUE),
          torch::nn_dropout(),
          torch::nn_linear(4096, num_classes),
        )

      },
      forward = function(x) {
        self$model(x)[,1]
      }
    )

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
          loss = nn_nll_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        fit(train_dl, epochs = 5, valid_data = valid_dl, verbose = TRUE)
    }
  }

  luz::luz_save(model, paste(as.character(getwd()), "model.pt", sep = "/"))

}
