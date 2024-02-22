#' @title Model Download

download_model <- function(url, model_name) {

  path <- getwd()

  download.file(url, paste(path, model_name, sep = "/"), mode = "wb")

  model_path <- paste(path, model_name, sep = "/")
}
