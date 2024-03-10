#' @title DenseNet
#' @import torch
#'
#' @export

# adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

.denselayer <- torch::nn_module(
  "denselayer",
  initialize = function(num_input_features, growth_rate, bn_size, drop_rate, memory_efficient = FALSE) {
    self$norm1 <- nn_batch_norm2d(num_input_features)
    self$relu1 <- nn_relu(inplace = TRUE)
    self$conv1 <- nn_conv2d(num_input_features, bn_size * growth_rate, kernel_size = 1, stride = 1, bias = FALSE)

    self$norm2 <- nn_batch_norm2d(bn_size * growth_rate)
    self$relu2 <- nn_relu(inplace = TRUE)
    self$conv2 <- nn_conv2d(bn_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = FALSE)

    self$drop_rate <-  as.double(drop_rate)
    self$memory_efficient <- memory_efficient
  },
  bn_function <- function(x) {
    concated_features <- torch::torch_cat(x, 1)
    bottleneck_output <- self$conv1(self$relu1(self$norm1(concated_features)))
    return(bottleneck_output)
  },

  forward <- function(x) {
    bottleneck_output <- self$bn_function(x)

    new_features <- self$conv2(self$relu2(self$norm2(bottleneck_output)))

    if (self$drop_rate > 0) {
      new_features <- nnf_dropout(new_features, p = self$drop_rate, training = self$training)
      return(new_features)
    }
  }
)

.denseblock <- torch::nn_module_dict(
  .version = 2,

  initialize <- function(num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient) {
    for (i in range(num_layers)) {
      layer <- .denselayer(
        num_input_features + i * growth_rate,
        growth_rate = growth_rate,
        bn_size = bn_size,
        drop_rate = drop_rate,
        memory_efficient = memory_efficient
      )

      self$.add_module((i + 1), layer)
    }
  },

  forward <- function(x) {
    features <- x

    for (layer in self$.items()) {
      new_features <- layer(features)
      append(features, new_features)
    }

    return(torch_cat(features, 1))
  }
)

.transition <- torch::nn_sequential(
  initialize <- function(num_input_features, num_output_features) {
    self$norm <- nn_batch_norm2d(num_input_features)
    self$relu <- nn_relu(inplace = TRUE)
    self$conv <- nn_conv2d(num_input_features, num_output_features, kernel_size = 1, stride = 1, bias = FALSE)
    self$pool <- nn_avg_pool2d(kernel_size = 2, stride = 2)
  }
)

densenet <- torch::nn_module(
  "DenseNet",
  initialize <- function(growth_rate = 32, block_config = list(6, 12, 24, 16), num_init_features = 64,
                         bn_size = 4, drop_rate = 0, num_classes = 1000, memory_efficient = FALSE) {

  }
)
