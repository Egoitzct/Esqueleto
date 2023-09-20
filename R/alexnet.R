#' @name alexnet
#' @title alexnet
#' @import torch
#' @import torchvision
#'
#' @export

## AlexNet implementation from https://github.com/mlverse/torchvision/blob/main/R/models-alexnet.R
alexnet_model <- torch::nn_module(
  "AlexNet",
  initialize = function(num_classes) {
    self$features <- torch::nn_sequential(
      torch::nn_conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(64, 192, kernel_size = 5, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(256, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2)
    )
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(6,6))
    self$classifier <- torch::nn_sequential(
      torch::nn_dropout(),
      torch::nn_linear(256 * 6 * 6, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_dropout(),
      torch::nn_linear(4096, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(4096, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    x
  }
)

alexnet <- function() {
  net <- alexnet_model
}
