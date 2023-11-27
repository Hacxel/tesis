# tesis
codigo utilizado en artículo para la mejora de pronósticos
# Librerías ---------------------------------------------------------------
library(xgboost)
library(tidymodels)
library(modeltime)
library(tidyverse)
library(glue)
library(lubridate)
library(timetk)
library(janitor)
library(ggpubr)
library(rio)

# Funciones personalizadas ------------------------------------------------

## Limpieza de la información ---------------------------------------------
getLimpiezaData <- function(.Data){
  ## Empresa 1: CI
  .Data <- clean_names(.Data, "big_camel") %>% 
    rename(Fecha = ends_with("FechaCorta"))
  
  # Formato Fecha
  dbDataClean <- .Data %>% 
    mutate(Mes = map_chr(Fecha, 
                         function(x){
                           mes = str_match(x, pattern = "/(\\w{1,2})/")[2]
                           mes = ifelse(str_length(mes)==1, paste("0", mes, sep = ""), mes)
                           
                           return(mes)
                         }),
           Fecha = pmap_chr(list(Fecha, Mes), 
                            function(Fecha, Mes){
                              
                              # Extracción del año
                              anio = str_match(Fecha, pattern = "(\\w{4})$")[1]
                              # Nueva fecha
                              NuevaFecha = paste(anio, Mes, "01", sep = "/")
                              
                              return(NuevaFecha)
                            }),
           Fecha = as.Date(Fecha)) %>% 
    select(-Mes)
  
  return(dbDataClean)
}


## Limpieza de la información ---------------------------------------------

## Pipeline Forcast -------------------------------------------------------


getForcastReport <- function(.Data, .Empresa = "", .VariableFecha, .VariableCantidad,
                             .MesesPrediccion = 3){
  
  # Renombre de las variables
  .Data <- .Data %>% 
    rename(Cantidad = .VariableCantidad,
           Fecha = .VariableFecha)
  
  # Parámetro plots de plotly (interactive) a ggplot (static)
  interactive <- FALSE
  
  # Split data --------------------------------------------------------------
  
  # Split Data 80/20
  cat("Split data (Entrenamiento 80% - Test 20%)\n"); cat("\n")
  splits <- initial_time_split(.Data, prop = 1 - (.MesesPrediccion/nrow(.Data)))
  
  # Modelos -----------------------------------------------------------------
  cat("Modelos forcast (8)\n"); cat("\n")
  
  ## Model 1: auto_arima ----
  # Auto ARIMA (Modeltime)
  cat("     (1/8) Auto ARIMA (Modeltime)\n")
  model_fit_arima_no_boost <- arima_reg() %>%
    set_engine(engine = "auto_arima") %>%
    fit(Cantidad ~ Fecha, data = training(splits))
  
  ## Model 2: arima_boost ----
  # Boosted Auto ARIMA (Modeltime)
  cat("     (2/8) Boosted Auto ARIMA (Modeltime)\n")
  model_fit_arima_boosted <- arima_boost(
    min_n = 2,
    learn_rate = 0.015
  ) %>%
    set_engine(engine = "auto_arima_xgboost") %>%
    fit(Cantidad ~ Fecha +
          as.numeric(Fecha) +
          factor(month(Fecha,    label = TRUE), ordered = F) +
          factor(quarter(Fecha), labels = TRUE, ordered = F),
        data = training(splits))
  
  
  ## Model 3: ets ----
  # Suavizado exponencial (Modeltime)
  cat("     (3/8) Suavizado exponencial (Modeltime)\n")
  model_fit_ets <- exp_smoothing() %>%
    set_engine(engine = "ets") %>%
    fit(Cantidad ~ Fecha, data = training(splits))
  
  
  ## Model 4: prophet ----
  # Prophet (Modeltime)
  cat("     (4/8) Prophet (Modeltime)\n")
  model_fit_prophet_no_boost <- prophet_reg() %>%
    set_engine(engine = "prophet") %>%
    fit(Cantidad ~ Fecha, data = training(splits))
  
  
  ## Model 5: Prophet Boost ----
  # Prophet Boost (Modeltime)
  cat("     (5/8) Prophet Boost (Modeltime)\n")
  model_fit_prophet_boost <- prophet_boost() %>%
    set_engine(engine = "prophet_xgboost") %>%
    fit(Cantidad ~ Fecha, data = training(splits))
  
  
  ## Model 6: lm ----
  # Linear Regression (Parsnip)
  cat("     (6/8) Linear Regression (Parsnip)\n")
  model_fit_lm <- linear_reg() %>%
    set_engine("lm") %>%
    fit(Cantidad ~ as.numeric(Fecha) +
          factor(month(Fecha,    label = TRUE), ordered = F) +
          factor(quarter(Fecha), labels = TRUE, ordered = F),
        data = training(splits))
  
  
  ## Model 7: earth ----
  # MARS (Workflow)
  cat("     (7/8) MARS (Workflow)\n")
  model_spec_mars <- mars(mode = "regression") %>%
    set_engine("earth") 
  
  recipe_spec <- recipe(Cantidad ~ Fecha, 
                        data = training(splits)) %>%
    step_date(Fecha, features = "month",   ordinal = FALSE) %>%
    step_date(Fecha, features = "quarter", ordinal = FALSE) %>%
    step_mutate(date_num = as.numeric(Fecha)) %>%
    step_normalize(date_num) %>%
    step_rm(Fecha)
  
  wflw_fit_mars <- workflow() %>%
    add_recipe(recipe_spec) %>%
    add_model(model_spec_mars) %>%
    fit(training(splits))
  
  
  ## Model 8: Random Forest ----
  # Random Forest (Workflow)
  cat("     (8/8) Random Forest (Workflow)\n"); cat("\n")
  model_spec_rf <- rand_forest(trees = 200) %>%
    set_engine("randomForest")
  
  wflw_fit_rf <- workflow() %>%
    add_recipe(recipe_spec) %>%
    add_model(model_spec_rf) %>%
    fit(training(splits))
  
  
  # 3. Tabla de modelos ------------------------------------------------------
  
  # Tabla de modelos a estimar
  models_tbl <- modeltime_table(
    model_fit_arima_no_boost,
    model_fit_arima_boosted,
    model_fit_ets,
    model_fit_prophet_no_boost,
    model_fit_prophet_boost,
    model_fit_lm,
    wflw_fit_mars,
    wflw_fit_rf
  )
  
  
  # 4. Calibración de los modelos -------------------------------------------
  cat("Calibracion de los modelos\n"); cat("\n")
  calibration_tbl <- models_tbl %>%
    modeltime_calibrate(new_data = testing(splits))
  
  
  # 5. Pronóstico del conjunto de pruebas y evaluación de la precisión --------
  
  # Plot forcasts
  (forcast_plot <- calibration_tbl %>%
     modeltime_forecast(
       new_data    = testing(splits),
       actual_data = as_tibble(.Data)
     ) %>%
     plot_modeltime_forecast(
       .legend_max_width = 25, # For mobile screens
       .interactive      = interactive
     ))
  
  # Table plot
  cat("Evaluacion de los modelos\n"); cat("\n")
  calibration_tbl %>%
    modeltime_accuracy() %>%
    table_modeltime_accuracy(
      .interactive = interactive
    )
  
  table_plot <- calibration_tbl %>%
    modeltime_accuracy() %>% 
    mutate_at(.vars = c("mae", "mape", "mase", "smape", "rmse", "rsq"), 
              .funs = function(x){round(x, 2)}) %>% 
    ggtexttable(rows = NULL, theme = ttheme("mBlueWhite")) %>% 
    tab_add_hline(at.row = 1:2, row.side = "top", linewidth = 2)
  
  # Plot results
  figure <- ggarrange(forcast_plot, table_plot,
                      ncol = 1, nrow = 2)
  
  plot(figure)
  
  # Save plot forcast test
  cat("Guardando reporte de resultados\n"); cat("\n")
  png(file= paste("reports/Resultados forcast empresa ", .Empresa, " ", Sys.Date(), " (",.MesesPrediccion, " meses).png", sep = ""),
      width=1400, height=750)
  plot(figure)
  dev.off()
  
  
  # 6. Forcast --------------------------------------------------------------
  
  # Modelo con menor error
  cat("Forcast\n"); cat("\n")
  MejorModelo <- calibration_tbl %>%
    modeltime_accuracy() %>% 
    mutate_at(.vars = c("mae", "mape", "mase", "smape", "rmse", "rsq"), 
              .funs = function(x){round(x, 2)}) %>% 
    slice_min(mape) %>% 
    slice_tail() %>% 
    pull(.model_id)
  
  MejorModeloNombre <- calibration_tbl %>%
    modeltime_accuracy() %>% 
    mutate_at(.vars = c("mae", "mape", "mase", "smape", "rmse", "rsq"), 
              .funs = function(x){round(x, 2)}) %>% 
    slice_min(mape) %>% 
    slice_tail() %>% 
    pull(.model_desc)
  
  cat(paste("Seleccion mejor modelo forcast:", MejorModeloNombre, "\n")); cat("\n")
  
  
  # Selección de estructura de datos para mejor modelo
  if(MejorModelo%in%c(1, 3, 4, 5, 7, 8)){
    dataForcast <- .Data
  } else if(MejorModelo==2){
    dataForcast <- .Data %>% 
      transmute(Cantidad, Fecha, as.numeric(Fecha),
                factor(month(Fecha, label = TRUE), ordered = F),
                factor(quarter(Fecha), labels = TRUE, ordered = F))
    
  } else if(MejorModelo==6){
    dataForcast <- .Data %>% 
      transmute(Cantidad, as.numeric(Fecha),
                factor(month(Fecha, label = TRUE), ordered = F),
                factor(quarter(Fecha), labels = TRUE, ordered = F))
  }
  
  # Forcast
  cat("Guardando resultados del Forcast\n"); cat("\n")
  ## Tabla
  tbForcast <- calibration_tbl %>%
    # Seleccion del mejor modelo
    filter(.model_id == MejorModelo) %>%
    # Refit and Forecast Forward
    modeltime_refit(.Data) %>%
    modeltime_forecast(h = paste(.MesesPrediccion, "months"), actual_data = .Data)
  
  ## Plot
  plot_forcast <- plot_modeltime_forecast(tbForcast, .interactive = FALSE)
  plot(plot_forcast)
  
  # Exportar forcast
  ## Save forcast
  tbForcastReport = select(tbForcast,
                           Dato = .key, Fecha = .index, Valor = .value) %>% 
    mutate(Dato = ifelse(Dato=="prediction", 'forcast', "dato"))
  
  export(tbForcastReport, 
         paste("forcast/Forcast empresa ", .Empresa, " ", Sys.Date(), " (", .MesesPrediccion, " meses).xlsx", sep = ""))
  
  ## Save plot forcast test
  png(file= paste("forcast/Forcast empresa ", .Empresa, " ", Sys.Date(), " (",.MesesPrediccion, " meses).png", sep = ""),
      width=1400, height=750)
  plot(plot_forcast)
  dev.off()
  
  # View plot results
  return(tbForcastReport)
}
