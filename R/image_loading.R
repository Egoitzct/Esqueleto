#' @title Image Loader
#' @import torch
#' @import torchvision
#' @import luz
#' @description
#' Una función para cargar una base de datos compuesta por imágenes. La base de datos deberá estar compuesta por tres carpetas (train, valid y test)
#'
#' @export

image_loading <- function(image_directory){
  if (torch::torch_is_installed() == FALSE) {
    torch::install_torch()
  }
  set.seed(123)
  torch::torch_manual_seed(123)

  train_transforms <- function(x) {
    x %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_random_resized_crop(size = c(64, 64), scale = c(0.96, 1), ratio = c(0.95, 1.05)) %>%
      torchvision::transform_color_jitter() %>%
      torchvision::transform_random_horizontal_flip() %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }

  valid_transforms <- function(x) {
    x %>%
      torchvision::transform_to_tensor() %>%
      torchvision::transform_resize(76) %>%
      torchvision::transform_center_crop(64) %>%
      torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
  }

  test_transforms <- valid_transforms

  train_ds <- torchvision::image_folder_dataset(
    file.path(image_directory, "train"),
    transform = train_transforms
  )

  test_ds <- torchvision::image_folder_dataset(
    file.path(image_directory, "test"),
    transform = test_transforms
  )

  valid_ds <- torchvision::image_folder_dataset(
    file.path(image_directory, "valid"),
    transform = valid_transforms
  )

  return(list(train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds))
}
