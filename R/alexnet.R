#' @title utils
#' @import torch


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
      torch::nn_dropout(p = 0.5),
      torch::nn_linear(256*6*6, 4096),
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
    x <- torch::torch_flatten(x, start_dim = 3)
    x <- self$classifier(x)
    x
  }
)

## ResNet implementation based on https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
conv3x3 <- function(in_planes, out_planes, stride=1, groups=1, dilation=1) {
  torch::nn_conv2d(
    in_planes,
    out_planes,
    kernel_size = 3,
    stride = stride,
    padding = dilation,
    groups = groups,
    bias = FALSE,
    dilation = dilation
  )
}

conv1x1 <- function(in_planes, out_planes, stride = 1) {
  torch::nn_conv2d(
    in_planes,
    out_planes,
    kernel_size = 1,
    stride = stride,
    bias = FALSE
  )
}

basic_block <- torch::nn_module(
  expansion <- 1,

  initialize = function(inplanes, planes, stride = 1, downsample = NULL, groups = 1, base_width = 64, dilation = 1, norm_layer = NULL) {
    if(is.null(norm_layer)) {norm_layer <- torch::nn_batch_norm2d}

    if (groups != 1 || base_width != 64) {rlang::abort("BasicBlock only supports groups = 1 and base_width = 64", class = "ValueError")}

    if (dilation > 1) {rlang::abort("Dilation > 1 not supported in BasicBlock", class = "NotImplementedError")}

    self$conv1 <- conv3x3(inplanes, planes, stride)
    self$bn1 <- norm_layer(planes)
    slef$relu <- torch::nn_relu(inplace = TRUE)
    self$conv2 <- conv3x3(planes, planes)
    self$bn2 <- norm_layer(planes)
    self$downsample <- downsample
    self$stride <- stride
  },

  forward = function(x) {
    identity <- x

    out <- self$conv1(x)
    out <- self$bn1(out)
    out <- self$relu(out)

    out <- self$conv2(out)
    out <- self$bn2(out)

    if (!is.null(self$downsample)) {
      identity <- self$downsample(x)
    } else {
      identity <- x
    }

    out <- out + identity
    out <- self$relu(out)

    out
  }
)

bottleneck <- torch::nn_module(
  expansion <-  4,

  initialize = function(inplanes, planes, stride = 1, downsample = NULL, groups = 1, base_width = 64, dilation = 1, norm_layer = NULL) {

    if (is.null(norm_layer)) {norm_layer <- torch::nn_batch_norm2d}

    width <- as.integer(planes * (base_width / 64)) * groups

    self$conv1 <- conv1x1(inplanes, width)
    self$bn1 <- norm_layer(width)
    self$conv2 <- conv3x3(width, width, stride, groups, dilation)
    self$bn2 <- norm_layer(width)
    self$conv3 <- conv1x1(width, planes * self$expansion)
    self$bn3 <- norm_layer(planes * self$expansion)
    self$relu <- torch::nn_relu(inplace = TRUE)
    self$downsample <- downsample
    self$stride <- stride
  },
  forward = function(x) {
    identity <- x

    out <- self$conv1(x)
    out <- self$bn1(out)
    out <- self$relu(out)

    out <- self$conv2(out)
    out <- self$bn2(out)
    out <- self$relu(out)

    out <- self$conv3(out)
    out <- self$bn3(out)

    if (!is.null(self$downsample)) {
      identity <- self$downsample(x)
    } else {
        identity <- x
      }

    out <- out + identity
    out <- self$relu(out)

    out
  }
)

resnet <- torch::nn_module(
  initialize = function(block, layers, num_classes, zero_init_residual = FALSE, groups = 1, width_per_group = 64, replace_stride_with_dilation = NULL, norm_layer = NULL) {

    if (is.null(norm_layer)) {norm_layer <- torch::nn_batch_norm2d}

    self$.norm_layer <- norm_layer

    self$inplanes <- 64
    self$dilation <- 1

    if (is.null(replace_stride_with_dilation)) {
      replace_stride_with_dilation <- rep(FALSE, 3)
    }

    if (length(replace_stride_with_dilation) != 3) {rlang::abort("replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")}

    self$groups <- groups
    self$base_width <- width_per_group
    self$conv1 <- torch::nn_conv2d(3, self$inplanes, kernel_size = 7, stride = 2, padding = 3, bias = FALSE)
    self$bn1 <- norm_layer(self$inplanes)
    self$relu <- torch::nn_relu(inplace = TRUE)
    self$maxpool <- torch::nn_max_pool2d(kernel_size = 3, stride = 2, padding = 1)
    self$layer1 <- self$.make_layer(block, 64, layers[1])
    self$layer2 <- self$.make_layer(block, 128, layers[2], stride = 2, dilate = replace_stride_with_dilation[1])
    self$layer3 <- self$.make_layer(block, 256, layers[3], stride = 2, dilate = replace_stride_with_dilation[2])
    self$layer4 <- self$.make_layer(block, 512, layers[4], stride = 2, dilate = replace_stride_with_dilation[3])
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- torch::nn_linear(512 * block$public_fields$expansion, num_classes) #"block$public_fields$expansion found at https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R

    for (m in private$modules_) {
      if (inherits(m, "nn_conv2d")) {
        torch::nn_init_kaiming_normal_(m$weight, mode="fan_out", nonlinearity = "relu")
      } else if (inherits(m, "nn_batch_norm2d") || inherits(m, "nn_group_norm")) {
          torch::nn_init_constant_(m$weight, 1)
          torch::nn_init_constant_(m$bias, 0)
        }
    }

    if (zero_init_residual) {
      for (m in private$modules_) {
        if (inherits(m, "bottleneck") && !is.null(m$bn3$weight)) {
          torch::nn_init_constant_(m$bn3$weight, 0)
        } else if (inherits(m, "basic_block") && !is.null(m$bn2$weight)) {
          torch::nn_init_constant_(m$bn2$weight, 0)
        }
      }
    }
  },
  .make_layer = function(block, planes, blocks, stride = 1, dilate = FALSE) {

    norm_layer <- self$.norm_layer
    downsample <- NULL
    previous_dilation <- self$dilation

    if (dilate) {
      self$dilation <- self$dilation * stride
      stride <- 1
    }

    if (stride != 1 || self$inplanes != planes * block$public_fields$expansion) {
      downsample <- torch::nn_sequential(
        conv1x1(self$inplanes, planes * block$public_fields$expansion, stride),
        norm_layer(planes * block$public_fields$expansion)
      )
    }

    layers <- list()
    layers[[1]] <- block(self$inplanes, planes, stride, downsample, self$groups, self$base_width, previous_dilation, norm_layer) #found at https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R

    self$inplanes <- planes * block$public_fields$expansion

    for (i in seq(from = 2, to = blocks)) {
      layers[[i]] <- block(self$inplanes, planes, groups = self$groups, base_width = self$base_width, dilation = self$dilation, norm_layer = norm_layer)
    }

    do.call(torch::nn_sequential, layers)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$bn1(x)
    x <- self$relu(x)
    x <- self$maxpool(x)

    x <- self$layer1(x)
    x <- self$layer2(x)
    x <- self$layer3(x)
    x <- self$layer4(x)

    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$fc(x)

    x
  }
)

.resnet <- function(block, layers, ...) { #implemented from https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R
  model <- resnet(block, layers, ...)

  model
}


alexnet <- function() {
  net <- alexnet_model
}

resnet18 <- function() {
  net <- .resnet(basic_block, c(2, 2, 2, 2))
}

resnet34 <- function() {
  net <- .resnet(basic_block, c(3, 4, 6, 3))
}

resnet50 <- function() {
  net <- .resnet(bottleneck, c(3, 4, 6, 3))
}

resnet101 <- function() {
  net <- .resnet(bottleneck, c(3, 4, 23, 3))
}

resnet152 <- function() {
  net <- .resnet(bottleneck, c(3, 8, 36, 3))
}

resnext50_32x4d <- function() {
  net <- .resnet(bottleneck, c(3, 4, 6, 3), groups = 32, width_per_group = 4)
}

resnext101_32x8d <- function(progress = TRUE) {
  net <- .resnet(bottleneck, c(3, 4, 23, 3), progress, groups = 32, width_per_group = 8)
}

wide_resnet50_2 <- function(progress = TRUE) {
  net <- .resnet(bottleneck, c(3, 4, 6, 3), progress, width_per_group = 64*2)
}

wide_resnet101_2 <- function(progress = TRUE) {
  net <- .resnet(bottleneck, c(3, 4, 23, 3), progress, width_per_group = 64*2)
}


