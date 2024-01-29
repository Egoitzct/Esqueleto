#' @title Entrenamiento de imágenes (pruebas)
#' @import torch
#' @import torchvision
#' @import luz
#'
#' @export

image_train_pruebas <- function(model = "alexnet", pretrained = FALSE, batch_size = 64) {
  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  # device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

  train_transforms <- function(x) {
    x %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_random_resized_crop(size = c(224, 224), scale = c(0.96, 1), ratio = c(0.95, 1.05)) %>%
      torchvision::transform_color_jitter() %>%
      torchvision::transform_random_horizontal_flip() %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }

  valid_transforms <- function(x) {
    x %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(256) %>%
      torchvision::transform_center_crop(224) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }

  test_transforms <- valid_transforms

  dir <- "~/.torch-datasets"

  train_ds <- tiny_imagenet_dataset(
    dir,
    download = TRUE,
    transform = train_transforms)

  valid_ds <- tiny_imagenet_dataset(
    dir,
    split = "val",
    transform = valid_transforms)

  train_dl <- dataloader(train_ds, batch_size = batch_size)
  valid_dl <- dataloader(valid_ds, batch_size = batch_size)

  num_classes <- length(train_ds$classes)

  if (model == "alexnet") {


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
        set_hparams(num_classes = 2) %>%
        fit(train_dl, epochs = 1, valid_data = valid_dl, verbose = TRUE) #Cambiar epochs después de pruebas
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
        fit(train_dl, epochs = 1, valid_data = valid_dl, verbose = TRUE)
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
