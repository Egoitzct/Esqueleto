#' @title Prediction
#' @param model_path Path to the trained model (the model must have been trained using the image_train function o using the "luz" package)
#' @param object_path Path to the object that is going to be classified
#' @description
#' A function used to predict the class of a single image using a previously trained model.
#'
#' @export
#'

prediction <- function(model_path, object_path){

  load_image <- function(path) { # https://github.com/mlverse/torchvision/blob/main/vignettes/examples/texture-nca.R line 152
    x <- torchvision::base_loader(path) %>%
      transform_to_tensor()
    x <- x[newaxis,..]
    x <- torchvision::transform_normalize(x, mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
    x <- torchvision::transform_resize(x, size = c(224, 224))
    x <- torchvision::transform_normalize(x, rep(0.5, 3), rep(0.5, 3))
  }

  loaded_object <- load_image(object_path)

  loaded_model <- luz::luz_load(model_path)

  signif(torch::as_array(torch_sigmoid(predict(loaded_model, loaded_object))), digits = 2)

}
