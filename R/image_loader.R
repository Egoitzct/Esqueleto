#' @title Image Loader
#' @export
#'

image_loader <- function(image_path, model, test = T){
  if (torch::torch_is_installed() == FALSE) {
    torch::install_torch()
  }
  set.seed(123)
  torch::torch_manual_seed(123)

  if (model == "alexnet") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_resized_crop(size = c(64, 64), scale = c(0.96, 1), ratio = c(0.95, 1.05)) %>%
        torchvision::transform_color_jitter() %>%
        torchvision::transform_random_horizontal_flip() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_resize(76) %>%
        torchvision::transform_center_crop(64) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  } else if (model == "resnet18") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  } else if (model == "resnet34") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  } else if (model == "resnet50") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  } else if (model == "resnet101") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  } else if (model == "resnet152") {
    train_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_random_affine(
          degrees = c(-30, 30), translate = c(0.2, 0.2)
        ) %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }
    valid_transforms = function(x) {
      x %>%
        torchvision::transform_to_tensor() %>%
        torchvision::transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    }

  }

  if (test == TRUE) {
    train_ds <- torchvision::image_folder_dataset(
      file.path(image_path, "train"),
      transform = train_transforms
    )

    test_ds <- torchvision::image_folder_dataset(
      file.path(image_path, "test"),
      transform = valid_transforms
    )

    valid_ds <- torchvision::image_folder_dataset(
      file.path(image_path, "valid"),
      transform = valid_transforms
    )

    return(list(train_ds = train_ds, valid_ds = valid_ds, test_ds = test_ds))

  } else {
    train_ds <- torchvision::image_folder_dataset(
      file.path(image_path, "train"),
      transform = train_transforms
    )

    valid_ds <- torchvision::image_folder_dataset(
      file.path(image_path, "valid"),
      transform = valid_transforms
    )

    return(list(train_ds = train_ds, valid_ds = valid_ds))
  }

}
