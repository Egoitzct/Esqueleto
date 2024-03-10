#' @title Entrenamiento de imagenes
#' @export
#'

entrenamiento_imagenes <- function(image_path, model = "alexnet", pretrained = F, batch_size = 128, epochs = 10, learning_rate = 0.01,
                                   optimizer = "adam"){
  if (torch::torch_is_installed() == FALSE) {

    torch::install_torch()
  }

  image_data <- image_loader(image_path, model)

  train_ds <- image_data$train_ds
  valid_ds <- image_data$valid_ds
  test_ds <- image_data$test_ds

  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE, drop_last = TRUE)
  valid_dl <- dataloader(valid_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)
  test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE, drop_last = TRUE)

  num_classes <- length(train_ds$classes)

  if (optimizer == "adam") {
    optimizador <- torch::optim_adam
  } else if (optimizer == "adadelta") {
    optimizador <- torch::optim_adadelta
  } else if (optimizer == "adagrad") {
    optimizador <- torch::optim_adagrad
  } else if (optimizer == "adamw") {
    optimizador <- torch::optim_adamw
  } else if (optimizer == "asgd") {
    optimizador <- torch::optim_asgd
  } else if (optimizer == "lbfgs") {
    optimizador <- torch::optim_lbfgs
  } else if (optimizer == "rmsprop") {
    optimizador <- torch::optim_rmsprop
  } else if (optimizer == "rprop") {
    optimizador <- torch::optim_rprop
  } else if (optimizer == "sgd") {
    optimizador <- torch::optim_sgd
  }

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
        self$model(x)
      }
    )

    if (num_classes == 0) {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  } else if (model == "resnet18") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- torchvision::model_resnet18(pretrained = pretrained)
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
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  } else if (model == "resnet34") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- torchvision::model_resnet34(pretrained = pretrained)
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
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  } else if (model == "resnet50") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- torchvision::model_resnet50(pretrained = pretrained)
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
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  } else if (model == "resnet101") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- torchvision::model_resnet101(pretrained = pretrained)
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
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  } else if (model == "resnet152") {
    net <- torch::nn_module(
      initialize = function(num_classes) {
        self$model <- torchvision::model_resnet152(pretrained = pretrained)
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
          loss = torch::nn_bce_with_logits_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_binary_accuracy_with_logits()
          )
        ) %>%
        set_hparams(num_classes = 1) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    } else {
      fitted_model <- net %>%
        setup(
          loss = torch::nn_cross_entropy_loss(),
          optimizer = optimizador,
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        set_hparams(num_classes = num_classes) %>%
        set_opt_hparams(lr = learning_rate) %>%
        fit(train_dl, epochs = epochs, valid_data = valid_dl, verbose = TRUE,
            callbacks = list(luz_callback_early_stopping(patience = 2)))
    }
  }

  model_name <- paste("trained", model, sep = "_")

  model_name <- paste(model_name, as.character(Sys.Date()), sep = "_")

  model_name <- paste(model_name, "pt", sep = ".")

  luz::luz_save(fitted_model, paste(as.character(getwd()), model_name, sep = "/"))

}
