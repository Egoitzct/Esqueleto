#' @title Entrenamiento de imágenes (pruebas)
#' @import torch
#' @import torchvision
#' @import luz
#'
#' @export

image_train_pruebas <- function(model = "alexnet", pretrained = FALSE, epochs = 10, batch_size = 32, datasets = "dogs_cats", image_path) {
  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  device <- "cpu"

  set.seed(123)
  torch_manual_seed(123)

  if (datasets == "dogs_cats") {

  dir <- "~/.torch-datasets/dogs-vs-cats"

  ds <- torchdatasets::dogs_vs_cats_dataset(
    dir,
    download = TRUE,
    transform = . %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(size = c(224, 224)) %>%
      torchvision::transform_normalize(rep(0.5, 3), rep(0.5, 3)),
    target_transform = function(x) as.double(x) - 1
  )

  train_id <- sample.int(length(ds), size = 0.7*length(ds))
  train_ds <- dataset_subset(ds, indices = train_id)
  valid_ds <- dataset_subset(ds, indices = which(!seq_along(ds) %in% train_id))

  train_dl <- dataloader(train_ds, batch_size = 64, shuffle = TRUE, num_workers = 4)
  valid_dl <- dataloader(valid_ds, batch_size = 64, num_workers = 4)

  } else if (datasets == "computer") {

    image_data <- image_loading(image_path)

    train_ds <- image_data$train_ds
    valid_ds <- image_data$valid_ds
    test_ds <- image_data$test_ds

    train_dl <- dataloader(train_ds, batch_size = batch_size)
    valid_dl <- dataloader(valid_ds, batch_size = batch_size)
    test_dl <- dataloader(test_ds, batch_size = batch_size)

  } else if (datasets == "mnist") {
    dir <- "~/Downloads/mnist" #caching directory

    train_ds <- mnist_dataset(
      dir,
      download = TRUE,
      transform = transform_to_tensor
    )

    test_ds <- mnist_dataset(
      dir,
      train = FALSE,
      transform = transform_to_tensor
    )

    train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)
    valid_dl <- dataloader(test_ds, batch_size = 32)

  } else if (datasets == "tinyimagenet") {
    dir <- "~/.torch-datasets/tiny-imagenet/"
    train_ds <- tiny_imagenet_dataset(
      dir,
      download = TRUE,
      transform = . %>%
        transform_to_tensor() %>%
        transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        transform_normalize(
          mean = c(0.485, 0.456, 0.406),
          std = c(0.229, 0.224, 0.225)
        )
    )

    valid_ds <- tiny_imagenet_dataset(
      dir,
      split = "val",
      transform = function(x) {
        x %>%
          transform_to_tensor() %>%
          transform_normalize(
            mean = c(0.485, 0.456, 0.406),
            std = c(0.229, 0.224, 0.225))
      }
    )

    train_dl <- dataloader(
      train_ds,
      batch_size = 128,
      shuffle = TRUE
    )
    valid_dl <- dataloader(valid_ds, batch_size = 128)
  } else if (datasets == "cifar10") {
    dir <- "~/.torch-datasets/cifar10/"
    train_ds <- cifar10_dataset(
      dir,
      download = TRUE,
      transform = function(x) {
        x %>%
          transform_to_tensor() %>%
          transform_resize(c(64, 64))
      }
    )

    valid_ds <- cifar10_dataset(
      dir,
      download = TRUE,
      train = FALSE,
      transform = function(x) {
        x %>%
          transform_to_tensor() %>%
          transform_resize(c(64,64))
      }
    )

    train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
    valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)
  }

  num_classes <- length(train_ds$classes)

  if (model == "alexnet") {

     net <- torch::nn_module(
       "AlexNet",
       initialize = function(num_classes) {
         self$model <- torchvision::model_alexnet(pretrained = pretrained)

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
      fitted_model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz::luz_callback_early_stopping()))
    } else {
      fitted_model <- net %>%
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
        self$model <- torchvision::model_vgg11(pretrained = pretrained)

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
      fitted_model <- net %>%
        setup(
          loss = nn_bce_with_logits_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        fit(train_dl, epochs = 2, valid_data = valid_dl, verbose = TRUE) #Cambiar epochs después de pruebas
    } else {
      fitted_model <- net %>%
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
  } else if (model == "resnet18") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- model_resnet18(pretrained = TRUE)
        for (par in self$parameters) {
          par$requires_grad_(FALSE)
        }
        self$model$fc <- nn_sequential(
          nn_linear(self$model$fc$in_features, 1024),
          nn_relu(),
          nn_linear(1024, 1024),
          nn_relu(),
          nn_linear(1024, num_classes)
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
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz::luz_callback_early_stopping()))
    } else {
      fitted_model <- net %>%
        setup(
          loss = nn_cross_entropy_loss(),
          optimizer = optim_adam,
          metrics = list(
            luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        fit(train_dl, epochs = 50, valid_data = valid_dl,
            callbacks = list(
              luz_callback_early_stopping(patience = 2),
              luz_callback_lr_scheduler(
                lr_one_cycle,
                max_lr = 0.01,
                epochs = 50,
                steps_per_epoch = length(train_dl),
                call_on = "on_batch_end"),
              luz_callback_model_checkpoint(path = "cpt_resnet/"),
              luz_callback_csv_logger("logs_resnet.csv")
            ),
            verbose = TRUE)
    }
  }

  model_name <- paste("trained", model, sep = "_")

  model_name <- paste(model_name, as.character(Sys.Date()), sep = "_")

  model_name <- paste(model_name, "pt", sep = ".")

  luz::luz_save(fitted_model, paste(as.character(getwd()), model_name, sep = "/"))

}
